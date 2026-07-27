import assert from "node:assert/strict"
import { describe, test } from "node:test"
import { parse } from "yaml"
import { parseObsidianSiteConfiguration, syncSiteConfigurationToQuartz } from "./obsidianSiteConfig"

const publishingReadme = `---
title: 发布配置
content_type: reference
---
# 发布配置

<!-- quartz-site-config -->
\`\`\`yaml
website:
  base_url: zhenwei.site
  branding:
    title: Zhenwei's Blog
    mark: Z
    home_label: Zhenwei's Blog 首页
    github_url: https://github.com/ZhenweiCao
    github_label: 在 GitHub 查看 Zhenwei Cao
  homepage:
    source: Website/首页.md
  assets:
    provider: tencent_cos
    object_prefix: zhenwei-site/assets
  navigation:
    pinned_limit: 2
    items:
      - label: 首页
        href: /
      - label: 知识库
        directory: GPU
        href: /gpu/index
\`\`\`
`

describe("Obsidian site configuration", () => {
  test("converts the marked body configuration to Quartz configuration", () => {
    assert.deepEqual(parseObsidianSiteConfiguration(publishingReadme), {
      baseUrl: "zhenwei.site",
      title: "Zhenwei's Blog",
      siteHeader: {
        mark: "Z",
        homeLabel: "Zhenwei's Blog 首页",
        githubUrl: "https://github.com/ZhenweiCao",
        githubLabel: "在 GitHub 查看 Zhenwei Cao",
      },
      homepage: {
        source: "Website/首页.md",
      },
      assets: {
        provider: "tencent_cos",
        objectPrefix: "zhenwei-site/assets",
      },
      navigation: {
        pinnedLimit: 2,
        pinnedItems: [
          { label: "首页", href: "/" },
          { label: "知识库", directory: "GPU", href: "/gpu/index" },
        ],
      },
    })
  })

  test("rejects navigation items without a directory or href", () => {
    const invalid = `---
title: 发布配置
---

<!-- quartz-site-config -->
\`\`\`yaml
website:
  base_url: zhenwei.site
  branding:
    title: Test
    mark: T
    home_label: Test 首页
  homepage:
    source: Website/首页.md
  navigation:
    pinned_limit: 1
    items:
      - label: 无链接
\`\`\`
`

    assert.throws(
      () => parseObsidianSiteConfiguration(invalid),
      /must define at least one of directory or href/,
    )
  })

  test("rejects machine-specific absolute directory mappings", () => {
    const invalid = `---
title: 发布配置
---

<!-- quartz-site-config -->
\`\`\`yaml
website:
  base_url: zhenwei.site
  branding:
    title: Test
    mark: T
    home_label: Test 首页
  homepage:
    source: Website/首页.md
  navigation:
    pinned_limit: 1
    items:
      - label: 知识库
        directory: /Users/someone/vault/GPU
\`\`\`
`

    assert.throws(() => parseObsidianSiteConfiguration(invalid), /must be relative/)
  })

  test("rejects base URLs with a protocol", () => {
    const invalid = publishingReadme.replace("zhenwei.site", "https://zhenwei.site")
    assert.throws(() => parseObsidianSiteConfiguration(invalid), /must not include a protocol/)
  })

  test("does not accept website configuration from document properties", () => {
    const propertiesOnly = `---
title: 发布配置
website:
  base_url: zhenwei.site
---
# 发布配置
`

    assert.throws(
      () => parseObsidianSiteConfiguration(propertiesOnly),
      /exactly one YAML block marked/,
    )
  })

  test("rejects multiple marked configuration blocks", () => {
    const duplicated = `${publishingReadme}
<!-- quartz-site-config -->
\`\`\`yaml
website: {}
\`\`\`
`

    assert.throws(() => parseObsidianSiteConfiguration(duplicated), /exactly one YAML block marked/)
  })

  test("updates only the managed website settings in Quartz YAML", () => {
    const quartzConfig = `configuration:
  pageTitle: Test
  baseUrl: old.example.com
  navigation:
    pinnedLimit: 1
    pinnedItems:
      - label: Old
        href: /old
plugins: []
`
    const siteConfiguration = parseObsidianSiteConfiguration(publishingReadme)
    const result = syncSiteConfigurationToQuartz(quartzConfig, siteConfiguration)
    const parsed = parse(result.output)

    assert.equal(result.changed, true)
    assert.equal(parsed.configuration.pageTitle, "Zhenwei's Blog")
    assert.equal(parsed.configuration.baseUrl, "zhenwei.site")
    assert.deepEqual(parsed.configuration.siteHeader, siteConfiguration.siteHeader)
    assert.deepEqual(parsed.configuration.navigation, siteConfiguration.navigation)
    assert.deepEqual(parsed.plugins, [])
  })

  test("reports no drift without rewriting an in-sync file", () => {
    const quartzConfig = `configuration:
  pageTitle: Zhenwei's Blog
  baseUrl: zhenwei.site
  siteHeader:
    mark: Z
    homeLabel: Zhenwei's Blog 首页
    githubUrl: https://github.com/ZhenweiCao
    githubLabel: 在 GitHub 查看 Zhenwei Cao
  navigation:
    pinnedLimit: 2
    pinnedItems:
      - label: 首页
        href: /
      - label: 知识库
        directory: GPU
        href: /gpu/index
`

    assert.deepEqual(
      syncSiteConfigurationToQuartz(quartzConfig, parseObsidianSiteConfiguration(publishingReadme)),
      {
        changed: false,
        output: quartzConfig,
      },
    )
  })
})
