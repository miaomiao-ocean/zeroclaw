use crate::providers::traits::{
    ChatMessage, ChatRequest as ProviderChatRequest, ChatResponse as ProviderChatResponse,
    NormalizedStopReason, Provider, ProviderCapabilities, StreamChunk, StreamError, StreamOptions,
    StreamResult, ToolCall as ProviderToolCall, ToolsPayload,
};
use crate::tools::ToolSpec;
use async_trait::async_trait;
use futures_util::stream::BoxStream;
use futures_util::{stream, StreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json as json;
use std::collections::HashMap;
use std::sync::LazyLock;

// ========================================================================
// 常量定义
// ========================================================================

/// DeepSeek API 基础 URL
const DEEPSEEK_BASE_URL: &str = "https://api.deepseek.com";
/// Reasoner 模型前缀（此类模型不支持 thinking 配置）
const REASONER_MODEL_PREFIX: &str = "reasoner";

/// 全局 HTTP client（带超时配置，使用 LazyLock 缓存以提高性能）
static HTTP_CLIENT: LazyLock<Client> = LazyLock::new(|| {
    crate::config::build_runtime_proxy_client_with_timeouts("provider.deepseek", 120, 10)
});

// ========================================================================
// 错误类型定义
// ========================================================================

/// DeepSeek API 错误
#[derive(Debug)]
pub struct DeepSeekError {
    pub message: String,
    pub request_id: Option<String>,
    pub status_code: Option<u16>,
}

impl DeepSeekError {
    /// 从响应创建错误
    pub fn from_response(
        status: reqwest::StatusCode,
        body: &str,
        request_id: Option<String>,
    ) -> Self {
        Self {
            message: format!("{}: {}", status.as_u16(), body),
            request_id,
            status_code: Some(status.as_u16()),
        }
    }

    /// 从请求错误创建
    pub fn from_request(error: reqwest::Error) -> Self {
        Self {
            message: format!("Request error: {}", error),
            request_id: None,
            status_code: None,
        }
    }

    /// 从 JSON 解析错误创建
    pub fn from_json_error(error: json::Error, request_id: Option<String>) -> Self {
        Self {
            message: format!("JSON parse error: {}", error),
            request_id,
            status_code: None,
        }
    }
}

impl std::fmt::Display for DeepSeekError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)?;
        if let Some(id) = &self.request_id {
            write!(f, " [request_id: {}]", id)?;
        }
        Ok(())
    }
}

impl std::error::Error for DeepSeekError {}

// ========================================================================
// 请求类型定义
// ========================================================================

/// DeepSeek API 聊天请求
#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<ThinkingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptionsPayload>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_logprobs: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logit_bias: Option<HashMap<String, f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,
}

/// Thinking 模式配置
#[derive(Debug, Serialize)]
struct ThinkingConfig {
    #[serde(rename = "type")]
    kind: String,
}

/// 响应格式配置
#[derive(Debug, Serialize)]
struct ResponseFormat {
    #[serde(rename = "type")]
    kind: String,
}

/// 工具选择策略
#[derive(Debug, Serialize)]
#[serde(untagged)]
enum ToolChoice {
    String(String),
    Object {
        #[serde(rename = "type")]
        tool_type: String,
        function: ToolChoiceFunction,
    },
}

impl ToolChoice {
    fn auto() -> Self {
        Self::String("auto".to_string())
    }
}

/// 工具选择函数
#[derive(Debug, Serialize)]
struct ToolChoiceFunction {
    name: String,
}

/// 流式选项载荷
#[derive(Debug, Serialize)]
struct StreamOptionsPayload {
    include_usage: bool,
}

/// 请求消息
#[derive(Debug, Serialize)]
struct Message {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefix: Option<bool>,
}

impl Message {
    /// 创建简单消息（无特殊字段）
    fn simple(role: &str, content: &str) -> Self {
        Self {
            role: role.to_string(),
            content: Some(content.to_string()),
            name: None,
            tool_call_id: None,
            tool_calls: None,
            reasoning_content: None,
            prefix: None,
        }
    }
}

// ========================================================================
// 响应类型定义
// ========================================================================

/// DeepSeek API 聊天响应
#[derive(Debug, Deserialize)]
struct ChatResponse {
    id: String,
    object: String,
    created: i64,
    model: String,
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Option<Usage>,
    #[serde(default)]
    system_fingerprint: Option<String>,
}

/// 响应选项
#[derive(Debug, Deserialize)]
struct Choice {
    index: u32,
    message: ResponseMessage,
    #[serde(default)]
    finish_reason: Option<FinishReason>,
    #[serde(default)]
    logprobs: Option<json::Value>,
}

/// 响应消息
#[derive(Debug, Deserialize)]
struct ResponseMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCall>>,
    role: Option<String>,
}

impl ResponseMessage {
    /// 获取有效内容（优先使用 content，否则使用 reasoning_content）
    fn effective_content(&self) -> Option<String> {
        self.content
            .as_ref()
            .filter(|c| !c.is_empty())
            .cloned()
            .or_else(|| self.reasoning_content.clone())
    }
}

/// 工具调用
#[derive(Debug, Deserialize, Serialize)]
struct ToolCall {
    id: Option<String>,
    #[serde(rename = "type", default)]
    kind: Option<String>,
    function: Option<Function>,
}

/// 函数调用参数
#[derive(Debug, Deserialize, Serialize)]
struct Function {
    name: Option<String>,
    arguments: Option<String>,
}

/// Token 使用统计
#[derive(Debug, Deserialize)]
struct Usage {
    #[serde(default)]
    prompt_tokens: Option<u32>,
    #[serde(default)]
    completion_tokens: Option<u32>,
    #[serde(default)]
    total_tokens: Option<u32>,
    #[serde(default)]
    prompt_cache_hit_tokens: Option<u32>,
    #[serde(default)]
    prompt_cache_miss_tokens: Option<u32>,
    #[serde(default)]
    completion_tokens_details: Option<CompletionTokensDetails>,
}

/// 完成令牌详情
#[derive(Debug, Deserialize)]
struct CompletionTokensDetails {
    #[serde(default)]
    reasoning_tokens: Option<u32>,
}

/// 响应完成原因
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    ContentFilter,
    ToolCalls,
    InsufficientSystemResource,
}

impl FinishReason {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "stop" => Some(Self::Stop),
            "length" => Some(Self::Length),
            "content_filter" => Some(Self::ContentFilter),
            "tool_calls" => Some(Self::ToolCalls),
            "insufficient_system_resource" => Some(Self::InsufficientSystemResource),
            _ => None,
        }
    }
}

impl<'de> Deserialize<'de> for FinishReason {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = Option::<String>::deserialize(deserializer)?;
        match s.as_deref().and_then(FinishReason::from_str) {
            Some(reason) => Ok(reason),
            None => Err(serde::de::Error::custom(format!(
                "invalid finish_reason: {:?}",
                s
            ))),
        }
    }
}

// ========================================================================
// 流式响应类型定义
// ========================================================================

/// 流式聊天响应块
#[derive(Debug, Deserialize)]
struct ChatResponseChunk {
    id: Option<String>,
    object: Option<String>,
    created: Option<i64>,
    model: Option<String>,
    choices: Vec<ChoiceChunk>,
    #[serde(default)]
    usage: Option<Usage>,
}

/// 流式响应选项
#[derive(Debug, Deserialize)]
struct ChoiceChunk {
    index: u32,
    delta: DeltaMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

/// 增量消息
#[derive(Debug, Deserialize)]
struct DeltaMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCall>>,
    #[serde(default)]
    role: Option<String>,
}

/// SSE 流式响应数据块
#[derive(Debug)]
struct ChunkData {
    content: Option<String>,
    reasoning: Option<String>,
}

// ========================================================================
// SSE 解析函数
// ========================================================================

/// 解析 SSE 行数据
fn parse_sse_line(line: &str) -> StreamResult<Option<ChunkData>> {
    let line = line.trim();

    // 跳过空行和注释
    if line.is_empty() || line.starts_with(':') {
        return Ok(None);
    }

    let Some(data) = line.strip_prefix("data:") else {
        return Ok(None);
    };

    let data = data.trim();

    // 流结束标记
    if data == "[DONE]" {
        return Ok(None);
    }

    // 解析 JSON
    let chunk: ChatResponseChunk = json::from_str(data).map_err(StreamError::Json)?;

    // 提取第一个 choice 的内容
    let Some(choice) = chunk.choices.first() else {
        return Ok(None);
    };

    let content = choice.delta.content.clone();
    let reasoning = choice.delta.reasoning_content.clone();

    // 只有当有实际内容时才返回
    if content.is_some() || reasoning.is_some() {
        return Ok(Some(ChunkData { content, reasoning }));
    }

    Ok(None)
}

// ========================================================================
// DeepSeek Provider 实现
// ========================================================================

/// DeepSeek 模型提供者
pub struct DeepSeekProvider {
    name: String,
    base_url: String,
    api_key: Option<String>,
    model: String,
    thinking_enabled: Option<bool>,
    json_output_enabled: bool,
    temperature: Option<f64>,
    max_tokens: Option<u32>,
    top_p: Option<f64>,
    frequency_penalty: Option<f64>,
    presence_penalty: Option<f64>,
    n: Option<u32>,
    logit_bias: Option<HashMap<String, f64>>,
    stop: Option<Vec<String>>,
    seed: Option<i64>,
    user: Option<String>,
}

impl DeepSeekProvider {
    /// 创建新的 DeepSeek 提供者实例
    pub fn new(api_key: Option<String>, model: String) -> Self {
        let thinking_enabled = if Self::is_reasoner_model(&model) {
            None
        } else {
            Some(false)
        };

        Self {
            name: "DeepSeek".to_string(),
            base_url: DEEPSEEK_BASE_URL.to_string(),
            api_key,
            model,
            thinking_enabled,
            json_output_enabled: false,
            temperature: None,
            max_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            n: None,
            logit_bias: None,
            stop: None,
            seed: None,
            user: None,
        }
    }

    /// 检查是否为 reasoner 模型
    fn is_reasoner_model(model: &str) -> bool {
        model.contains(REASONER_MODEL_PREFIX)
    }

    /// 获取 HTTP client（使用全局缓存）
    fn http_client(&self) -> &Client {
        &HTTP_CLIENT
    }

    /// 获取 thinking 配置（reasoner 模型返回 None）
    fn thinking_config(&self) -> Option<ThinkingConfig> {
        if Self::is_reasoner_model(&self.model) {
            return None;
        }
        self.thinking_enabled.map(|enabled| ThinkingConfig {
            kind: if enabled { "enabled" } else { "disabled" }.to_string(),
        })
    }

    /// 检查是否需要禁用某些参数
    fn temperature_params_disabled(&self) -> bool {
        Self::is_reasoner_model(&self.model) || self.thinking_enabled == Some(true)
    }

    /// 获取响应格式配置
    fn response_format(&self) -> Option<ResponseFormat> {
        self.json_output_enabled.then_some(ResponseFormat {
            kind: "json_object".to_string(),
        })
    }

    /// 转换工具规范
    fn convert_tools(&self, tools: Option<&[ToolSpec]>) -> Option<Vec<json::Value>> {
        tools.map(|ts| {
            ts.iter()
                .map(|t| {
                    json::json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                        }
                    })
                })
                .collect()
        })
    }

    /// 将通用消息格式转换为 DeepSeek API 格式
    fn convert_messages(&self, messages: &[ChatMessage]) -> Vec<Message> {
        messages.iter().map(|m| self.convert_message(m)).collect()
    }

    /// 转换单条消息
    fn convert_message(&self, m: &ChatMessage) -> Message {
        match m.role.as_str() {
            "assistant" => self.convert_assistant_message(&m.content),
            "tool" => self.convert_tool_message(&m.content),
            _ => Message::simple(&m.role, &m.content),
        }
    }

    /// 从 JSON 中提取字符串字段
    fn extract_string_field(value: &json::Value, field: &str) -> Option<String> {
        value
            .get(field)
            .and_then(json::Value::as_str)
            .map(String::from)
    }

    /// 尝试将内容解析为 JSON，失败时返回原始字符串
    fn parse_content_json(&self, content: &str) -> Option<json::Value> {
        json::from_str::<json::Value>(content).ok()
    }

    /// 转换 assistant 消息（可能包含 tool_calls）
    fn convert_assistant_message(&self, content: &str) -> Message {
        let Some(value) = self.parse_content_json(content) else {
            return Message::simple("assistant", content);
        };

        let tool_calls = value
            .get("tool_calls")
            .and_then(|v| json::from_value::<Vec<ProviderToolCall>>(v.clone()).ok())
            .map(|calls| {
                calls
                    .into_iter()
                    .map(|tc| ToolCall {
                        id: Some(tc.id),
                        kind: Some("function".to_string()),
                        function: Some(Function {
                            name: Some(tc.name),
                            arguments: Some(tc.arguments),
                        }),
                    })
                    .collect::<Vec<_>>()
            });

        Message {
            role: "assistant".to_string(),
            content: Self::extract_string_field(&value, "content"),
            name: None,
            tool_call_id: None,
            tool_calls,
            reasoning_content: None,
            prefix: None,
        }
    }

    /// 转换 tool 消息
    fn convert_tool_message(&self, content: &str) -> Message {
        let Some(value) = self.parse_content_json(content) else {
            return Message::simple("tool", content);
        };

        Message {
            role: "tool".to_string(),
            content: Self::extract_string_field(&value, "content"),
            name: None,
            tool_call_id: Self::extract_string_field(&value, "tool_call_id"),
            tool_calls: None,
            reasoning_content: None,
            prefix: None,
        }
    }

    /// 构建聊天请求
    fn build_chat_request(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[ToolSpec]>,
        stream: Option<bool>,
        stream_options: Option<StreamOptions>,
    ) -> ChatRequest {
        let params_disabled = self.temperature_params_disabled();

        // 根据参数是否被禁用来选择使用哪个值
        let temperature = if params_disabled {
            None
        } else {
            self.temperature
        };
        let top_p = if params_disabled { None } else { self.top_p };
        let frequency_penalty = if params_disabled {
            None
        } else {
            self.frequency_penalty
        };
        let presence_penalty = if params_disabled {
            None
        } else {
            self.presence_penalty
        };

        let stream_enabled = stream.unwrap_or(false);
        let tools_enabled = tools.is_some();

        ChatRequest {
            model: self.model.clone(),
            messages: self.convert_messages(messages),
            thinking: self.thinking_config(),
            response_format: self.response_format(),
            tools: self.convert_tools(tools),
            tool_choice: if tools_enabled {
                Some(ToolChoice::auto())
            } else {
                None
            },
            temperature,
            max_tokens: self.max_tokens,
            top_p,
            frequency_penalty,
            presence_penalty,
            stream,
            stream_options: if stream_enabled {
                stream_options.map(|opts| StreamOptionsPayload {
                    include_usage: opts.count_tokens,
                })
            } else {
                None
            },
            stop: self.stop.clone(),
            logprobs: None,
            top_logprobs: None,
            n: self.n,
            logit_bias: self.logit_bias.clone(),
            seed: self.seed,
            user: self.user.clone(),
        }
    }

    /// 发送聊天请求
    async fn send_chat_request(
        &self,
        request: &ChatRequest,
    ) -> Result<ChatResponse, DeepSeekError> {
        let client = self.http_client();
        let url = format!("{}/chat/completions", self.base_url);

        let mut req_builder = client.post(&url);

        if let Some(key) = &self.api_key {
            req_builder = req_builder.header("Authorization", format!("Bearer {}", key));
        }

        let response = req_builder
            .json(request)
            .send()
            .await
            .map_err(DeepSeekError::from_request)?;

        if !response.status().is_success() {
            let status = response.status();
            // 尝试从响应头提取 request_id
            let request_id = response
                .headers()
                .get("x-request-id")
                .and_then(|v| v.to_str().ok())
                .map(String::from);
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| format!("HTTP error: {}", status));
            return Err(DeepSeekError::from_response(
                status,
                &error_text,
                request_id,
            ));
        }

        response.json().await.map_err(|e| DeepSeekError {
            message: format!("Failed to parse response: {}", e),
            request_id: None,
            status_code: None,
        })
    }

    /// 从 API 工具调用转换为内部工具调用
    fn convert_api_tool_calls(tool_calls: Option<Vec<ToolCall>>) -> Vec<ProviderToolCall> {
        tool_calls
            .unwrap_or_default()
            .into_iter()
            .filter_map(|tc| {
                let function = tc.function?;
                Some(ProviderToolCall {
                    id: tc.id?,
                    name: function.name?,
                    arguments: function.arguments?,
                })
            })
            .collect()
    }

    /// 从 API 响应中提取数据
    fn extract_response_data(
        &self,
        chat_resp: ChatResponse,
    ) -> Result<
        (
            Option<String>,
            Vec<ProviderToolCall>,
            Option<String>,
            Option<FinishReason>,
            Option<crate::providers::traits::TokenUsage>,
        ),
        DeepSeekError,
    > {
        let choice = chat_resp
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| DeepSeekError {
                message: "No choices in response".to_string(),
                request_id: None,
                status_code: None,
            })?;

        let message = choice.message;
        let text = message.effective_content();
        let finish_reason = choice.finish_reason;
        let tool_calls = Self::convert_api_tool_calls(message.tool_calls);
        let reasoning = message.reasoning_content;

        let usage = chat_resp
            .usage
            .map(|u| crate::providers::traits::TokenUsage {
                input_tokens: u.prompt_tokens.map(u32::into),
                output_tokens: u.completion_tokens.map(u32::into),
            });

        Ok((text, tool_calls, reasoning, finish_reason, usage))
    }

    /// 将 FinishReason 转换为标准化的停止原因
    fn convert_stop_reason(
        finish_reason: Option<FinishReason>,
    ) -> (Option<NormalizedStopReason>, Option<String>) {
        let reason = finish_reason.map(|f| match f {
            FinishReason::Stop => NormalizedStopReason::EndTurn,
            FinishReason::Length => NormalizedStopReason::MaxTokens,
            FinishReason::ContentFilter => NormalizedStopReason::SafetyBlocked,
            FinishReason::ToolCalls => NormalizedStopReason::ToolCall,
            FinishReason::InsufficientSystemResource => {
                NormalizedStopReason::Unknown("insufficient_system_resource".to_string())
            }
        });
        let raw = finish_reason.map(|f| format!("{:?}", f).to_lowercase());
        (reason, raw)
    }

    /// 流式聊天内部实现
    fn stream_chat_internal(
        &self,
        messages: &[ChatMessage],
        options: StreamOptions,
    ) -> BoxStream<'static, StreamResult<StreamChunk>> {
        let api_key = match &self.api_key {
            Some(key) => key.clone(),
            None => {
                let chunk = StreamChunk::error("DeepSeek API key not set");
                return stream::once(async move { Ok(chunk) }).boxed();
            }
        };

        let request = self.build_chat_request(messages, None, Some(true), Some(options));
        let url = format!("{}/chat/completions", self.base_url);
        let client = self.http_client().clone();

        let (tx, rx) = tokio::sync::mpsc::channel::<StreamResult<StreamChunk>>(256);

        tokio::spawn(async move {
            // 发送请求
            let response = match client
                .post(&url)
                .json(&request)
                .header("Authorization", format!("Bearer {}", api_key))
                .header("Accept", "text/event-stream")
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    let _ = tx.send(Err(StreamError::Http(e))).await;
                    return;
                }
            };

            // 检查响应状态
            if !response.status().is_success() {
                let status = response.status();
                let error = response
                    .text()
                    .await
                    .unwrap_or_else(|_| format!("HTTP error: {}", status));
                let _ = tx
                    .send(Err(StreamError::Provider(format!("{}: {}", status, error))))
                    .await;
                return;
            }

            // 处理 SSE 流
            let mut buffer = String::new();
            let mut bytes_stream = response.bytes_stream();

            while let Some(item) = bytes_stream.next().await {
                let bytes = match item {
                    Ok(b) => b,
                    Err(e) => {
                        let _ = tx.send(Err(StreamError::Http(e))).await;
                        break;
                    }
                };

                let text = match String::from_utf8(bytes.to_vec()) {
                    Ok(t) => t,
                    Err(e) => {
                        let _ = tx
                            .send(Err(StreamError::InvalidSse(format!(
                                "Invalid UTF-8: {}",
                                e
                            ))))
                            .await;
                        break;
                    }
                };

                buffer.push_str(&text);

                // 处理缓冲区中的每一行
                while let Some(pos) = buffer.find('\n') {
                    let line = buffer.drain(..=pos).collect::<String>();

                    match parse_sse_line(&line) {
                        Ok(Some(chunk_data)) => {
                            // 优先使用 content，为空时 fallback 到 reasoning_content
                            let text = chunk_data
                                .content
                                .filter(|c| !c.is_empty())
                                .or_else(|| chunk_data.reasoning);

                            if let Some(text) = text {
                                let chunk = StreamChunk::delta(text);
                                if tx.send(Ok(chunk)).await.is_err() {
                                    return;
                                }
                            }
                        }
                        Ok(None) => {}
                        Err(e) => {
                            let _ = tx.send(Err(e)).await;
                            return;
                        }
                    }
                }
            }

            let _ = tx.send(Ok(StreamChunk::final_chunk())).await;
        });

        stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|chunk| (chunk, rx))
        })
        .boxed()
    }
}

// ========================================================================
// Builder 方法实现
// ========================================================================

impl DeepSeekProvider {
    /// 设置自定义 API 基础 URL
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// 设置模型名称
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        let model_str = model.into();
        // 切换到 reasoner 模型时清除 thinking 配置
        if Self::is_reasoner_model(&model_str) {
            self.thinking_enabled = None;
        }
        self.model = model_str;
        self
    }

    /// 设置 thinking 模式（仅对非 reasoner 模型有效）
    fn set_thinking(&mut self, enabled: bool) {
        if !Self::is_reasoner_model(&self.model) {
            self.thinking_enabled = Some(enabled);
        }
    }

    /// 启用 thinking 模式
    pub fn enable_thinking(mut self) -> Self {
        self.set_thinking(true);
        self
    }

    /// 禁用 thinking 模式
    pub fn disable_thinking(mut self) -> Self {
        self.set_thinking(false);
        self
    }

    /// 启用 JSON 输出
    pub fn enable_json_output(mut self) -> Self {
        self.json_output_enabled = true;
        self
    }

    /// 设置温度
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature.clamp(0.0, 2.0));
        self
    }

    /// 设置最大令牌数
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// 设置 top_p
    pub fn with_top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p.clamp(0.0, 1.0));
        self
    }

    /// 设置频率惩罚
    pub fn with_frequency_penalty(mut self, penalty: f64) -> Self {
        self.frequency_penalty = Some(penalty.clamp(-2.0, 2.0));
        self
    }

    /// 设置存在惩罚
    pub fn with_presence_penalty(mut self, penalty: f64) -> Self {
        self.presence_penalty = Some(penalty.clamp(-2.0, 2.0));
        self
    }

    /// 设置生成多少个完成结果（n 参数）
    pub fn with_n(mut self, n: u32) -> Self {
        self.n = Some(n.max(1).min(10));
        self
    }

    /// 设置 logit_bias（token ID 到偏置值的映射）
    pub fn with_logit_bias(mut self, logit_bias: HashMap<String, f64>) -> Self {
        self.logit_bias = Some(logit_bias);
        self
    }

    /// 添加单个 stop 序列
    pub fn with_stop(mut self, stop: impl Into<String>) -> Self {
        let stop_str = stop.into();
        if let Some(ref mut stops) = self.stop {
            if stops.len() < 4 {
                stops.push(stop_str);
            }
        } else if self.stop.is_none() {
            self.stop = Some(vec![stop_str]);
        }
        self
    }

    /// 设置 stop 序列（最多4个）
    pub fn with_stops(mut self, stops: Vec<String>) -> Self {
        self.stop = Some(stops.into_iter().take(4).collect());
        self
    }

    /// 设置 seed（确定性采样）
    pub fn with_seed(mut self, seed: i64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// 设置 user（用户标识）
    pub fn with_user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }
}

// ========================================================================
// Provider Trait 实现
// ========================================================================

#[async_trait]
impl Provider for DeepSeekProvider {
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            native_tool_calling: true,
            vision: false,
        }
    }

    fn convert_tools(&self, tools: &[ToolSpec]) -> ToolsPayload {
        let tools_payload = self.convert_tools(Some(tools)).unwrap_or_default();
        ToolsPayload::OpenAI {
            tools: tools_payload,
        }
    }

    fn supports_native_tools(&self) -> bool {
        true
    }

    async fn warmup(&self) -> anyhow::Result<()> {
        let messages = vec![ChatMessage::user("ping")];
        let req = self.build_chat_request(&messages, None, None, None);
        let _ = self.send_chat_request(&req).await;
        Ok(())
    }

    async fn chat_with_system(
        &self,
        system_prompt: Option<&str>,
        message: &str,
        _model: &str,
        _temperature: f64,
    ) -> anyhow::Result<String> {
        let messages: Vec<ChatMessage> = if let Some(sys) = system_prompt {
            vec![ChatMessage::system(sys), ChatMessage::user(message)]
        } else {
            vec![ChatMessage::user(message)]
        };

        let req = self.build_chat_request(&messages, None, None, None);
        let chat_resp = self.send_chat_request(&req).await?;
        let (text, _, _, _, _) = self.extract_response_data(chat_resp)?;

        text.ok_or_else(|| anyhow::anyhow!("No content in response"))
    }

    async fn chat_with_history(
        &self,
        messages: &[ChatMessage],
        _model: &str,
        _temperature: f64,
    ) -> anyhow::Result<String> {
        let req = self.build_chat_request(messages, None, None, None);
        let chat_resp = self.send_chat_request(&req).await?;
        let (text, _, _, _, _) = self.extract_response_data(chat_resp)?;

        text.ok_or_else(|| anyhow::anyhow!("No content in response"))
    }

    async fn chat(
        &self,
        request: ProviderChatRequest<'_>,
        _model: &str,
        _temperature: f64,
    ) -> anyhow::Result<ProviderChatResponse> {
        let req = self.build_chat_request(request.messages, request.tools, None, None);
        let chat_resp = self.send_chat_request(&req).await?;
        let (text, tool_calls, reasoning, finish_reason, usage) =
            self.extract_response_data(chat_resp)?;

        let (stop_reason, raw_stop_reason) = Self::convert_stop_reason(finish_reason);

        Ok(ProviderChatResponse {
            text,
            tool_calls,
            usage,
            reasoning_content: reasoning,
            quota_metadata: None,
            stop_reason,
            raw_stop_reason,
        })
    }

    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[serde_json::Value],
        _model: &str,
        _temperature: f64,
    ) -> anyhow::Result<ProviderChatResponse> {
        let tool_specs: Vec<ToolSpec> = tools
            .iter()
            .filter_map(|t| {
                let function = t.get("function")?;
                Some(ToolSpec {
                    name: function.get("name")?.as_str()?.to_string(),
                    description: function
                        .get("description")
                        .and_then(|d| d.as_str())
                        .unwrap_or("")
                        .to_string(),
                    parameters: function
                        .get("parameters")
                        .cloned()
                        .unwrap_or(json::json!({})),
                })
            })
            .collect();

        let req = self.build_chat_request(messages, Some(&tool_specs), None, None);
        let chat_resp = self.send_chat_request(&req).await?;
        let (text, tool_calls, reasoning, finish_reason, usage) =
            self.extract_response_data(chat_resp)?;

        let (stop_reason, raw_stop_reason) = Self::convert_stop_reason(finish_reason);

        Ok(ProviderChatResponse {
            text,
            tool_calls,
            usage,
            reasoning_content: reasoning,
            quota_metadata: None,
            stop_reason,
            raw_stop_reason,
        })
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn stream_chat_with_system(
        &self,
        system_prompt: Option<&str>,
        message: &str,
        _model: &str,
        _temperature: f64,
        options: StreamOptions,
    ) -> BoxStream<'static, StreamResult<StreamChunk>> {
        let mut messages = Vec::new();
        if let Some(sys) = system_prompt {
            messages.push(ChatMessage::system(sys));
        }
        messages.push(ChatMessage::user(message));

        self.stream_chat_internal(&messages, options)
    }

    fn stream_chat_with_history(
        &self,
        messages: &[ChatMessage],
        _model: &str,
        _temperature: f64,
        options: StreamOptions,
    ) -> BoxStream<'static, StreamResult<StreamChunk>> {
        self.stream_chat_internal(messages, options)
    }
}

// ========================================================================
// 单元测试
// ========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_construction() {
        let provider =
            DeepSeekProvider::new(Some("test-key".to_string()), "deepseek-chat".to_string());
        assert_eq!(provider.model, "deepseek-chat");
        assert_eq!(provider.thinking_enabled, Some(false));
        assert_eq!(provider.base_url, DEEPSEEK_BASE_URL);
        assert!(!provider.json_output_enabled);
    }

    #[test]
    fn test_provider_construction_no_api_key() {
        let provider = DeepSeekProvider::new(None, "deepseek-chat".to_string());
        assert!(provider.api_key.is_none());
        assert_eq!(provider.model, "deepseek-chat");
    }

    #[test]
    fn test_reasoner_model_no_thinking_config() {
        let provider = DeepSeekProvider::new(
            Some("test-key".to_string()),
            "deepseek-reasoner".to_string(),
        );
        assert_eq!(provider.model, "deepseek-reasoner");
        assert!(provider.thinking_enabled.is_none());
    }

    #[test]
    fn test_enable_thinking() {
        let provider =
            DeepSeekProvider::new(Some("test-key".to_string()), "deepseek-chat".to_string())
                .enable_thinking();
        assert_eq!(provider.thinking_enabled, Some(true));
    }

    #[test]
    fn test_disable_thinking() {
        let provider =
            DeepSeekProvider::new(Some("test-key".to_string()), "deepseek-chat".to_string())
                .enable_thinking()
                .disable_thinking();
        assert_eq!(provider.thinking_enabled, Some(false));
    }

    #[test]
    fn test_enable_thinking_ignored_for_reasoner() {
        let provider = DeepSeekProvider::new(
            Some("test-key".to_string()),
            "deepseek-reasoner".to_string(),
        )
        .enable_thinking();
        assert!(provider.thinking_enabled.is_none());
    }

    #[test]
    fn test_disable_thinking_ignored_for_reasoner() {
        let provider = DeepSeekProvider::new(
            Some("test-key".to_string()),
            "deepseek-reasoner".to_string(),
        )
        .disable_thinking();
        assert!(provider.thinking_enabled.is_none());
    }

    #[test]
    fn test_with_base_url() {
        let provider = DeepSeekProvider::new(None, "deepseek-chat".to_string())
            .with_base_url("https://custom.api.com");
        assert_eq!(provider.base_url, "https://custom.api.com");
    }

    #[test]
    fn test_with_model() {
        let provider = DeepSeekProvider::new(None, "deepseek-chat".to_string())
            .with_model("deepseek-reasoner");
        assert_eq!(provider.model, "deepseek-reasoner");
        assert!(provider.thinking_enabled.is_none());
    }

    #[test]
    fn test_temperature_clamping() {
        let provider =
            DeepSeekProvider::new(None, "deepseek-chat".to_string()).with_temperature(3.0);
        assert_eq!(provider.temperature, Some(2.0));

        let provider =
            DeepSeekProvider::new(None, "deepseek-chat".to_string()).with_temperature(-1.0);
        assert_eq!(provider.temperature, Some(0.0));

        let provider =
            DeepSeekProvider::new(None, "deepseek-chat".to_string()).with_temperature(1.0);
        assert_eq!(provider.temperature, Some(1.0));
    }

    #[test]
    fn test_top_p_clamping() {
        let provider = DeepSeekProvider::new(None, "deepseek-chat".to_string()).with_top_p(1.5);
        assert_eq!(provider.top_p, Some(1.0));

        let provider = DeepSeekProvider::new(None, "deepseek-chat".to_string()).with_top_p(-0.5);
        assert_eq!(provider.top_p, Some(0.0));

        let provider = DeepSeekProvider::new(None, "deepseek-chat".to_string()).with_top_p(0.5);
        assert_eq!(provider.top_p, Some(0.5));
    }

    #[test]
    fn test_penalty_clamping() {
        let provider =
            DeepSeekProvider::new(None, "deepseek-chat".to_string()).with_frequency_penalty(3.0);
        assert_eq!(provider.frequency_penalty, Some(2.0));

        let provider =
            DeepSeekProvider::new(None, "deepseek-chat".to_string()).with_presence_penalty(-3.0);
        assert_eq!(provider.presence_penalty, Some(-2.0));

        let provider =
            DeepSeekProvider::new(None, "deepseek-chat".to_string()).with_frequency_penalty(0.5);
        assert_eq!(provider.frequency_penalty, Some(0.5));

        let provider =
            DeepSeekProvider::new(None, "deepseek-chat".to_string()).with_presence_penalty(-0.5);
        assert_eq!(provider.presence_penalty, Some(-0.5));
    }

    #[test]
    fn test_enable_json_output() {
        let provider =
            DeepSeekProvider::new(None, "deepseek-chat".to_string()).enable_json_output();
        assert!(provider.json_output_enabled);
    }

    #[test]
    fn test_with_max_tokens() {
        let provider =
            DeepSeekProvider::new(None, "deepseek-chat".to_string()).with_max_tokens(4096);
        assert_eq!(provider.max_tokens, Some(4096));
    }

    #[test]
    fn test_thinking_config_for_chat_model() {
        let provider = DeepSeekProvider::new(None, "deepseek-chat".to_string()).enable_thinking();
        let config = provider.thinking_config().unwrap();
        assert_eq!(config.kind, "enabled");

        let provider = DeepSeekProvider::new(None, "deepseek-chat".to_string()).disable_thinking();
        let config = provider.thinking_config().unwrap();
        assert_eq!(config.kind, "disabled");
    }

    #[test]
    fn test_thinking_config_for_reasoner_model() {
        let provider = DeepSeekProvider::new(None, "deepseek-reasoner".to_string());
        assert!(provider.thinking_config().is_none());
    }

    #[test]
    fn test_response_format_when_enabled() {
        let provider =
            DeepSeekProvider::new(None, "deepseek-chat".to_string()).enable_json_output();
        let format = provider.response_format().unwrap();
        assert_eq!(format.kind, "json_object");
    }

    #[test]
    fn test_response_format_when_disabled() {
        let provider = DeepSeekProvider::new(None, "deepseek-chat".to_string());
        assert!(provider.response_format().is_none());
    }

    #[test]
    fn test_convert_messages() {
        let provider = DeepSeekProvider::new(None, "deepseek-chat".to_string());
        let messages = vec![
            ChatMessage::system("You are helpful"),
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi there"),
        ];
        let converted = provider.convert_messages(&messages);

        assert_eq!(converted.len(), 3);
        assert_eq!(converted[0].role, "system");
        assert_eq!(converted[0].content, Some("You are helpful".to_string()));
        assert_eq!(converted[1].role, "user");
        assert_eq!(converted[1].content, Some("Hello".to_string()));
        assert_eq!(converted[2].role, "assistant");
        assert_eq!(converted[2].content, Some("Hi there".to_string()));
    }

    #[test]
    fn test_convert_tools() {
        let provider = DeepSeekProvider::new(None, "deepseek-chat".to_string());
        let tools = vec![ToolSpec {
            name: "get_weather".to_string(),
            description: "Get weather info".to_string(),
            parameters: json::json!({"type": "object"}),
        }];
        let converted = provider.convert_tools(Some(&tools)).unwrap();

        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0]["type"], "function");
        assert_eq!(converted[0]["function"]["name"], "get_weather");
        assert_eq!(converted[0]["function"]["description"], "Get weather info");
    }

    #[test]
    fn test_convert_tools_none() {
        let provider = DeepSeekProvider::new(None, "deepseek-chat".to_string());
        assert!(provider.convert_tools(None).is_none());
    }

    #[test]
    fn test_convert_tools_empty() {
        let provider = DeepSeekProvider::new(None, "deepseek-chat".to_string());
        let tools: Vec<ToolSpec> = vec![];
        let converted = provider.convert_tools(Some(&tools)).unwrap();
        assert!(converted.is_empty());
    }

    #[test]
    fn test_effective_content_prefers_content() {
        let msg = ResponseMessage {
            content: Some("hello".to_string()),
            reasoning_content: Some("thinking".to_string()),
            tool_calls: None,
            role: Some("assistant".to_string()),
        };
        assert_eq!(msg.effective_content(), Some("hello".to_string()));
    }

    #[test]
    fn test_effective_content_falls_back_to_reasoning() {
        let msg = ResponseMessage {
            content: Some(String::new()),
            reasoning_content: Some("thinking".to_string()),
            tool_calls: None,
            role: Some("assistant".to_string()),
        };
        assert_eq!(msg.effective_content(), Some("thinking".to_string()));
    }

    #[test]
    fn test_effective_content_both_none() {
        let msg = ResponseMessage {
            content: None,
            reasoning_content: None,
            tool_calls: None,
            role: Some("assistant".to_string()),
        };
        assert!(msg.effective_content().is_none());
    }

    #[test]
    fn test_effective_content_only_reasoning() {
        let msg = ResponseMessage {
            content: None,
            reasoning_content: Some("reasoning only".to_string()),
            tool_calls: None,
            role: Some("assistant".to_string()),
        };
        assert_eq!(msg.effective_content(), Some("reasoning only".to_string()));
    }

    #[test]
    fn test_finish_reason_from_str() {
        assert_eq!(FinishReason::from_str("stop"), Some(FinishReason::Stop));
        assert_eq!(FinishReason::from_str("length"), Some(FinishReason::Length));
        assert_eq!(
            FinishReason::from_str("content_filter"),
            Some(FinishReason::ContentFilter)
        );
        assert_eq!(
            FinishReason::from_str("tool_calls"),
            Some(FinishReason::ToolCalls)
        );
        assert_eq!(
            FinishReason::from_str("insufficient_system_resource"),
            Some(FinishReason::InsufficientSystemResource)
        );
        assert_eq!(FinishReason::from_str("unknown"), None);
        assert_eq!(FinishReason::from_str(""), None);
    }

    #[test]
    fn test_finish_reason_deserialize() {
        let json = r#""stop""#;
        let reason: FinishReason = json::from_str(json).unwrap();
        assert_eq!(reason, FinishReason::Stop);

        let json = r#""tool_calls""#;
        let reason: FinishReason = json::from_str(json).unwrap();
        assert_eq!(reason, FinishReason::ToolCalls);
    }

    #[test]
    fn test_finish_reason_deserialize_invalid() {
        let json = r#""invalid_reason""#;
        let result: Result<FinishReason, _> = json::from_str(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_chat_response_deserialization() {
        let json = r#"{
            "id": "test-id",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "deepseek-chat",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello, world!",
                    "reasoning_content": "Let me think..."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }"#;

        let response: ChatResponse = json::from_str(json).unwrap();
        assert_eq!(response.id, "test-id");
        assert_eq!(response.object, "chat.completion");
        assert_eq!(response.created, 1234567890);
        assert_eq!(response.model, "deepseek-chat");
        assert_eq!(response.choices.len(), 1);
        assert_eq!(
            response.choices[0].message.content,
            Some("Hello, world!".to_string())
        );
        assert_eq!(
            response.choices[0].message.reasoning_content,
            Some("Let me think...".to_string())
        );
        assert_eq!(response.choices[0].finish_reason, Some(FinishReason::Stop));
    }

    #[test]
    fn test_chat_response_with_tool_calls() {
        let json = r#"{
            "id": "test-id",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "deepseek-chat",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"location\": \"Beijing\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }"#;

        let response: ChatResponse = json::from_str(json).unwrap();
        let tool_calls = response.choices[0].message.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, Some("call_123".to_string()));
        assert_eq!(
            tool_calls[0].function.as_ref().unwrap().name,
            Some("get_weather".to_string())
        );
        assert_eq!(
            response.choices[0].finish_reason,
            Some(FinishReason::ToolCalls)
        );
    }

    #[test]
    fn test_chat_response_chunk_deserialization() {
        let json = r#"{
            "id": "chunk-id",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "deepseek-chat",
            "choices": [{
                "index": 0,
                "delta": {
                    "content": "hello",
                    "reasoning_content": "thinking"
                },
                "finish_reason": null
            }]
        }"#;

        let chunk: ChatResponseChunk = json::from_str(json).unwrap();
        assert_eq!(chunk.id, Some("chunk-id".to_string()));
        let delta = &chunk.choices[0].delta;
        assert_eq!(delta.content, Some("hello".to_string()));
        assert_eq!(delta.reasoning_content, Some("thinking".to_string()));
        assert!(chunk.choices[0].finish_reason.is_none());
    }

    #[test]
    fn test_parse_sse_line_done() {
        let result = parse_sse_line("data: [DONE]").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_sse_line_empty() {
        let result = parse_sse_line("").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_sse_line_whitespace() {
        let result = parse_sse_line("   ").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_sse_line_comment() {
        let result = parse_sse_line(": comment").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_sse_line_with_content() {
        let json = r#"data: {"choices":[{"index":0,"delta":{"content":"test"}}]}"#;
        let result = parse_sse_line(json).unwrap();
        assert!(result.is_some());
        let chunk = result.unwrap();
        assert_eq!(chunk.content, Some("test".to_string()));
        assert!(chunk.reasoning.is_none());
    }

    #[test]
    fn test_parse_sse_line_with_reasoning() {
        let json = r#"data: {"choices":[{"index":0,"delta":{"reasoning_content":"thinking..."}}]}"#;
        let result = parse_sse_line(json).unwrap();
        assert!(result.is_some());
        let chunk = result.unwrap();
        assert!(chunk.content.is_none());
        assert_eq!(chunk.reasoning, Some("thinking...".to_string()));
    }

    #[test]
    fn test_parse_sse_line_with_both() {
        let json = r#"data: {"choices":[{"index":0,"delta":{"content":"answer","reasoning_content":"thought"}}]}"#;
        let result = parse_sse_line(json).unwrap();
        assert!(result.is_some());
        let chunk = result.unwrap();
        assert_eq!(chunk.content, Some("answer".to_string()));
        assert_eq!(chunk.reasoning, Some("thought".to_string()));
    }

    #[test]
    fn test_parse_sse_line_empty_delta() {
        let json = r#"data: {"choices":[{"index":0,"delta":{}}]}"#;
        let result = parse_sse_line(json).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_sse_line_invalid_json() {
        let result = parse_sse_line("data: {invalid}");
        assert!(result.is_err());
    }

    #[test]
    fn test_build_chat_request_basic() {
        let provider = DeepSeekProvider::new(None, "deepseek-chat".to_string());
        let messages = vec![ChatMessage::user("Hello")];
        let request = provider.build_chat_request(&messages, None, None, None);

        assert_eq!(request.model, "deepseek-chat");
        assert_eq!(request.messages.len(), 1);
        assert!(request.thinking.is_some());
        assert_eq!(request.thinking.as_ref().unwrap().kind, "disabled");
        assert!(request.tools.is_none());
        assert!(request.stream.is_none());
    }

    #[test]
    fn test_build_chat_request_with_stream_options() {
        let provider = DeepSeekProvider::new(None, "deepseek-chat".to_string());
        let messages = vec![ChatMessage::user("Hello")];
        let stream_opts = StreamOptions::new(true).with_token_count();
        let request = provider.build_chat_request(&messages, None, Some(true), Some(stream_opts));

        assert_eq!(request.stream, Some(true));
        assert!(request.stream_options.is_some());
        assert!(request.stream_options.unwrap().include_usage);
    }

    #[test]
    fn test_build_chat_request_includes_all_fields() {
        let provider = DeepSeekProvider::new(Some("key".to_string()), "deepseek-chat".to_string())
            .with_temperature(0.7)
            .with_max_tokens(1000)
            .with_top_p(0.9)
            .with_frequency_penalty(0.5)
            .with_presence_penalty(0.3)
            .enable_json_output();

        let messages = vec![ChatMessage::user("Hello")];
        let request = provider.build_chat_request(&messages, None, None, None);

        assert_eq!(request.model, "deepseek-chat");
        assert_eq!(request.temperature, Some(0.7));
        assert_eq!(request.max_tokens, Some(1000));
        assert_eq!(request.top_p, Some(0.9));
        assert_eq!(request.frequency_penalty, Some(0.5));
        assert_eq!(request.presence_penalty, Some(0.3));
        assert!(request.response_format.is_some());
        assert!(request.thinking.is_some());
    }

    #[test]
    fn test_build_chat_request_thinking_disables_temperature_params() {
        let provider = DeepSeekProvider::new(Some("key".to_string()), "deepseek-chat".to_string())
            .with_temperature(0.7)
            .with_max_tokens(1000)
            .with_top_p(0.9)
            .with_frequency_penalty(0.5)
            .with_presence_penalty(0.3)
            .enable_json_output()
            .enable_thinking();

        let messages = vec![ChatMessage::user("Hello")];
        let request = provider.build_chat_request(&messages, None, None, None);

        assert_eq!(request.model, "deepseek-chat");
        // Thinking 模式下这些参数应该被禁用
        assert_eq!(request.temperature, None);
        assert_eq!(request.top_p, None);
        assert_eq!(request.frequency_penalty, None);
        assert_eq!(request.presence_penalty, None);
        assert_eq!(request.max_tokens, Some(1000));
        assert!(request.response_format.is_some());
        assert!(request.thinking.is_some());
    }

    #[test]
    fn test_build_chat_request_reasoner_model_disables_temperature_params() {
        let provider =
            DeepSeekProvider::new(Some("key".to_string()), "deepseek-reasoner".to_string())
                .with_temperature(0.7)
                .with_max_tokens(1000)
                .with_top_p(0.9)
                .with_frequency_penalty(0.5)
                .with_presence_penalty(0.3)
                .enable_json_output();

        let messages = vec![ChatMessage::user("Hello")];
        let request = provider.build_chat_request(&messages, None, None, None);

        assert_eq!(request.model, "deepseek-reasoner");
        // Reasoner 模式下这些参数应该被禁用
        assert_eq!(request.temperature, None);
        assert_eq!(request.top_p, None);
        assert_eq!(request.frequency_penalty, None);
        assert_eq!(request.presence_penalty, None);
        assert_eq!(request.max_tokens, Some(1000));
        assert!(request.response_format.is_some());
        assert!(request.thinking.is_none());
    }

    #[test]
    fn test_build_chat_request_with_tools() {
        let provider = DeepSeekProvider::new(None, "deepseek-chat".to_string());
        let tools = vec![ToolSpec {
            name: "test".to_string(),
            description: "test tool".to_string(),
            parameters: json::json!({"type": "object"}),
        }];
        let messages = vec![ChatMessage::user("Hello")];
        let request = provider.build_chat_request(&messages, Some(&tools), None, None);

        assert!(request.tools.is_some());
        assert!(request.tool_choice.is_some());
    }

    #[test]
    fn test_tool_choice_serialization() {
        let choice = ToolChoice::auto();
        let json = json::to_string(&choice).unwrap();
        assert!(json.contains("auto"));
    }

    #[test]
    fn test_error_display() {
        let err = DeepSeekError {
            message: "Not found".to_string(),
            request_id: Some("req-123".to_string()),
            status_code: Some(404),
        };
        let display = err.to_string();
        assert!(display.contains("Not found"));
        assert!(display.contains("req-123"));
    }

    #[test]
    fn test_capabilities() {
        let provider = DeepSeekProvider::new(None, "deepseek-chat".to_string());
        let caps = provider.capabilities();
        assert!(caps.native_tool_calling);
        assert!(!caps.vision);
    }

    #[test]
    fn test_convert_tools_returns_openai_format() {
        let provider = DeepSeekProvider::new(None, "deepseek-chat".to_string());
        let tools = vec![ToolSpec {
            name: "get_weather".to_string(),
            description: "Get weather info".to_string(),
            parameters: json::json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                }
            }),
        }];
        let payload = DeepSeekProvider::convert_tools(&provider, Some(&tools));
        assert!(payload.is_some());
        let tools_json = payload.unwrap();
        assert_eq!(tools_json.len(), 1);
        assert_eq!(tools_json[0]["type"], "function");
        assert_eq!(tools_json[0]["function"]["name"], "get_weather");
    }

    #[test]
    fn test_with_n() {
        let provider = DeepSeekProvider::new(None, "deepseek-chat".to_string()).with_n(3);
        assert_eq!(provider.n, Some(3));

        let provider = DeepSeekProvider::new(None, "deepseek-chat".to_string()).with_n(15);
        assert_eq!(provider.n, Some(10));
    }

    #[test]
    fn test_with_logit_bias() {
        let mut bias = HashMap::new();
        bias.insert("12345".to_string(), -5.0);
        bias.insert("67890".to_string(), 2.0);

        let provider =
            DeepSeekProvider::new(None, "deepseek-chat".to_string()).with_logit_bias(bias.clone());

        assert_eq!(provider.logit_bias, Some(bias));
    }

    #[test]
    fn test_with_stop() {
        let provider = DeepSeekProvider::new(None, "deepseek-chat".to_string())
            .with_stop("STOP")
            .with_stop("END");

        assert_eq!(
            provider.stop,
            Some(vec!["STOP".to_string(), "END".to_string()])
        );
    }

    #[test]
    fn test_with_stops() {
        let provider = DeepSeekProvider::new(None, "deepseek-chat".to_string()).with_stops(vec![
            "STOP1".to_string(),
            "STOP2".to_string(),
            "STOP3".to_string(),
        ]);

        assert_eq!(
            provider.stop,
            Some(vec![
                "STOP1".to_string(),
                "STOP2".to_string(),
                "STOP3".to_string()
            ])
        );
    }

    #[test]
    fn test_with_stops_max_four() {
        let provider = DeepSeekProvider::new(None, "deepseek-chat".to_string()).with_stops(vec![
            "STOP1".to_string(),
            "STOP2".to_string(),
            "STOP3".to_string(),
            "STOP4".to_string(),
            "STOP5".to_string(),
        ]);

        assert_eq!(provider.stop.unwrap().len(), 4);
    }

    #[test]
    fn test_build_chat_request_with_new_params() {
        let mut bias = HashMap::new();
        bias.insert("123".to_string(), 1.0);

        let provider = DeepSeekProvider::new(Some("key".to_string()), "deepseek-chat".to_string())
            .with_n(2)
            .with_logit_bias(bias)
            .with_stop("END");

        let messages = vec![ChatMessage::user("Hello")];
        let request = provider.build_chat_request(&messages, None, None, None);

        assert_eq!(request.n, Some(2));
        assert!(request.logit_bias.is_some());
        assert_eq!(request.stop, Some(vec!["END".to_string()]));
    }
}
