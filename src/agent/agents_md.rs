use crate::config::DelegateAgentConfig;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tracing::warn;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarkdownAgent {
    pub name: String,
    pub config: DelegateAgentConfig,
    pub file_path: PathBuf,
}

#[derive(thiserror::Error, Debug)]
pub enum AgentLoadError {
    #[error("Failed to parse YAML: {0}")]
    YamlParseError(String),
    #[error("Invalid format: {0}")]
    FormatError(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

impl MarkdownAgent {
    pub fn parse(
        content: &str,
        default_provider: &str,
        default_model: &str,
    ) -> Result<Self, AgentLoadError> {
        let trimmed = content.trim();

        if !trimmed.starts_with("---") {
            return Err(AgentLoadError::FormatError(
                "Missing opening --- in YAML frontmatter".to_string(),
            ));
        }

        let after_first_dash = &trimmed[3..];

        if let Some(end_pos) = after_first_dash.find("\n---") {
            let yaml_content = after_first_dash[..end_pos].trim();

            if yaml_content.is_empty() {
                return Err(AgentLoadError::FormatError(
                    "Empty YAML frontmatter".to_string(),
                ));
            }

            #[derive(Deserialize)]
            struct Frontmatter {
                name: String,
                #[serde(default)]
                description: String,
                #[serde(default = "default_mode")]
                mode: String,
                #[serde(default)]
                provider: Option<String>,
                #[serde(default)]
                model: Option<String>,
                #[serde(default)]
                temperature: Option<f64>,
                #[serde(default)]
                api_key: Option<String>,
                #[serde(default = "default_max_depth")]
                max_depth: u32,
                #[serde(default)]
                agentic: bool,
                #[serde(default)]
                allowed_tools: Vec<String>,
                #[serde(default = "default_max_iterations")]
                max_iterations: usize,
            }

            let fm: Frontmatter = serde_yaml::from_str(yaml_content)
                .map_err(|e| AgentLoadError::YamlParseError(e.to_string()))?;

            if let Some(temp) = fm.temperature {
                if !(0.0..=2.0).contains(&temp) {
                    return Err(AgentLoadError::FormatError(
                        "temperature must be between 0.0 and 2.0".to_string(),
                    ));
                }
            }

            let prompt_start = 3 + end_pos + 4;
            let prompt = trimmed[prompt_start..].trim().to_string();

            let system_prompt = if fm.description.is_empty() && prompt.is_empty() {
                None
            } else if fm.description.is_empty() {
                Some(prompt)
            } else if prompt.is_empty() {
                Some(fm.description)
            } else {
                Some(format!("{}\n\n{}", fm.description, prompt))
            };

            return Ok(MarkdownAgent {
                name: fm.name,
                config: DelegateAgentConfig {
                    provider: fm.provider.unwrap_or_else(|| default_provider.to_string()),
                    model: fm.model.unwrap_or_else(|| default_model.to_string()),
                    system_prompt,
                    api_key: fm.api_key,
                    temperature: fm.temperature,
                    max_depth: fm.max_depth,
                    agentic: fm.agentic || fm.mode == "subagent",
                    allowed_tools: fm.allowed_tools,
                    max_iterations: fm.max_iterations,
                },
                file_path: PathBuf::new(),
            });
        }

        Err(AgentLoadError::FormatError(
            "Missing closing --- in YAML frontmatter".to_string(),
        ))
    }

    pub fn with_path(mut self, path: PathBuf) -> Self {
        self.file_path = path;
        self
    }
}

fn default_mode() -> String {
    "assistant".to_string()
}

fn default_max_depth() -> u32 {
    3
}

fn default_max_iterations() -> usize {
    10
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEFAULT_PROVIDER: &str = "openrouter";
    const DEFAULT_MODEL: &str = "anthropic/claude-sonnet-4.6";

    #[test]
    fn test_parse_agent_config() {
        let content = r#"---
name: test-agent
description: A test agent
mode: subagent
temperature: 0.5
---

You are a test agent."#;

        let agent = MarkdownAgent::parse(content, DEFAULT_PROVIDER, DEFAULT_MODEL).unwrap();
        assert_eq!(agent.name, "test-agent");
        assert_eq!(agent.config.temperature, Some(0.5));
        assert!(agent.config.agentic);
    }

    #[test]
    fn test_parse_agent_config_with_prompt() {
        let content = r#"---
name: test-agent
description: A test agent
mode: subagent
temperature: 0.5
---

You are a test agent.
You should respond with hello."#;

        let agent = MarkdownAgent::parse(content, DEFAULT_PROVIDER, DEFAULT_MODEL).unwrap();
        assert_eq!(agent.name, "test-agent");
        assert_eq!(agent.config.model, DEFAULT_MODEL);
        assert!(agent.config.system_prompt.is_some());
    }

    #[test]
    fn test_parse_missing_frontmatter() {
        let content = "You are a test agent.";
        let result = MarkdownAgent::parse(content, DEFAULT_PROVIDER, DEFAULT_MODEL);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_empty_frontmatter() {
        let content = r#"---
---

You are a test agent."#;
        let result = MarkdownAgent::parse(content, DEFAULT_PROVIDER, DEFAULT_MODEL);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_missing_name() {
        let content = r#"---
description: A test agent
mode: subagent
temperature: 0.5
---

You are a test agent."#;
        let result = MarkdownAgent::parse(content, DEFAULT_PROVIDER, DEFAULT_MODEL);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_invalid_temperature() {
        let content = r#"---
name: test-agent
description: A test agent
mode: subagent
temperature: not-a-number
---

You are a test agent."#;
        let result = MarkdownAgent::parse(content, DEFAULT_PROVIDER, DEFAULT_MODEL);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_temperature_out_of_range() {
        let content = r#"---
name: test-agent
description: A test agent
mode: subagent
temperature: 3.0
---

You are a test agent."#;
        let result = MarkdownAgent::parse(content, DEFAULT_PROVIDER, DEFAULT_MODEL);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_agents_empty_directory() {
        let temp_dir = tempfile::tempdir().unwrap();
        let agents_dir = temp_dir.path();

        let result = load_agents(agents_dir, DEFAULT_PROVIDER, DEFAULT_MODEL);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_load_agents_nonexistent_directory() {
        let temp_dir = tempfile::tempdir().unwrap();
        let agents_dir = temp_dir.path().join("nonexistent");

        let result = load_agents(&agents_dir, DEFAULT_PROVIDER, DEFAULT_MODEL);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_load_agents_multiple_files() {
        let temp_dir = tempfile::tempdir().unwrap();
        let agents_dir = temp_dir.path();
        std::fs::create_dir_all(agents_dir).unwrap();

        let content1 = r#"---
name: agent-one
description: First agent
mode: subagent
temperature: 0.5
---

You are agent one."#;
        let content2 = r#"---
name: agent-two
description: Second agent
mode: assistant
temperature: 0.7
---

You are agent two."#;
        let content3 = r#"---
name: agent-three
description: Third agent
mode: subagent
temperature: 0.3
---

You are agent three."#;

        std::fs::write(agents_dir.join("agent1.md"), content1).unwrap();
        std::fs::write(agents_dir.join("agent2.md"), content2).unwrap();
        std::fs::write(agents_dir.join("agent3.md"), content3).unwrap();
        std::fs::write(agents_dir.join("readme.txt"), "not an agent").unwrap();

        let result = load_agents(agents_dir, DEFAULT_PROVIDER, DEFAULT_MODEL).unwrap();
        assert_eq!(result.len(), 3);

        let names: Vec<_> = result.iter().map(|a| a.name.clone()).collect();
        assert!(names.contains(&"agent-one".to_string()));
        assert!(names.contains(&"agent-two".to_string()));
        assert!(names.contains(&"agent-three".to_string()));
    }

    #[test]
    fn test_load_agent_invalid_yaml() {
        let temp_dir = tempfile::tempdir().unwrap();
        let agents_dir = temp_dir.path();
        std::fs::create_dir_all(agents_dir).unwrap();

        std::fs::write(agents_dir.join("bad.md"), "not valid yaml ---").unwrap();

        let result = load_agents(agents_dir, DEFAULT_PROVIDER, DEFAULT_MODEL);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }
}

pub fn load_agents(
    agents_dir: &Path,
    default_provider: &str,
    default_model: &str,
) -> Result<Vec<MarkdownAgent>, AgentLoadError> {
    if !agents_dir.exists() {
        std::fs::create_dir_all(agents_dir)?;
        return Ok(vec![]);
    }

    let entries = std::fs::read_dir(agents_dir)?;

    let mut agents = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("md") {
            match load_agent(&path, default_provider, default_model) {
                Ok(agent) => agents.push(agent),
                Err(e) => warn!("Failed to load agent from {:?}: {}", path, e),
            }
        }
    }

    Ok(agents)
}

fn load_agent(
    file_path: &Path,
    default_provider: &str,
    default_model: &str,
) -> Result<MarkdownAgent, AgentLoadError> {
    let content = std::fs::read_to_string(file_path)?;
    let mut agent = MarkdownAgent::parse(&content, default_provider, default_model)?;
    agent.file_path = file_path.to_path_buf();
    Ok(agent)
}
