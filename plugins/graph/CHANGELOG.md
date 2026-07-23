# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-07-23

### Changed

- Keep local graphs focused on the current page.
- Cap dense local graphs while prioritizing outgoing links.
- Deduplicate bidirectional visual edges.
- Hide tag pages when tag nodes are disabled.
- Match the blog's visual language.
- Add a persistent, accessible switch for the graph panel.
- Pause graph rendering in reader mode and restore it when leaving.
- Cancel stale Pixi/D3 renders during navigation, collapse, and modal close.
- Load graph libraries only when the expanded panel needs them.
- Tolerate unavailable or corrupt browser storage.
