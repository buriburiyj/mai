---
name: project-overview
description: Inspect a software project using safe file-listing and file-reading tools, then explain its architecture and important files.
compatibility: MAI workspace with fs_list and fs_read access.
allowed-tools: fs_list fs_read
metadata:
  author: mai
  version: "1"
  auto-triggers: "project structure architecture codebase modules files directories repository 프로젝트 구조 아키텍처 코드베이스 모듈 파일 디렉터리 저장소"
---

# Project overview

Use this skill when the user asks about a project's structure or architecture.

1. Call `fs_list` on the workspace or requested directory.
2. Identify important source, test, configuration, and documentation files.
3. Read only the files necessary to understand the architecture.
4. Do not claim that a file exists unless it appeared in tool output.
5. Summarize:
   - project purpose
   - entry points
   - core modules
   - tests
   - security boundaries
   - suggested next steps
6. Mention any uncertainty or files that were not inspected.
7. MCP always means Model Context Protocol in this project; never expand it to another phrase.
