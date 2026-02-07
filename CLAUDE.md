# 核心规则 (Core Rules)

- **极度简洁 (Extreme Concision)**：
  在所有的交互、计划和提交信息（Commit Messages）中，务必保持极度简洁。为了达到简洁的目的，甚至可以牺牲语法的准确性。（严禁废话）。

- **GitHub CLI 优先 (GitHub CLI First)**：
  你与 GitHub 交互的首选方式应是使用 `gh` CLI 工具（当需要特定 GitHub 功能时，优先于直接使用 git 命令）。

- **分支命名 (Branch Naming)**：
  创建分支时，始终添加前缀 `cc/`，以表明该分支是由本次 AI 会话创建的。

- **规划协议 (Planning Protocol)**：
  1. **先计划，后编码**：在编写任何代码之前，必须先创建一个计划。
  2. **关键步骤 (Crucial)**：在每个计划阶段结束时，必须显式列出需要我回答的 **“未决问题” (Unresolved Questions)**。
  3. **保持简短**：这些问题必须保持极度精简。

- **项目上下文 (Project Context)**：
  - 我想学习这个仓库对应的代码