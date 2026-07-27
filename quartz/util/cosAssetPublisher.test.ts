import assert from "node:assert/strict"
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises"
import { join } from "node:path"
import { tmpdir } from "node:os"
import { after, before, describe, test } from "node:test"
import {
  createDeterministicAssetPublisher,
  createVaultAssetIndex,
  rewriteMarkdownAssetLinks,
} from "./cosAssetPublisher"

let vaultRoot: string

before(async () => {
  vaultRoot = await mkdtemp(join(tmpdir(), "quartz-cos-assets-"))
  await mkdir(join(vaultRoot, "GPU", "Drawings"), { recursive: true })
  await mkdir(join(vaultRoot, "Attachment"), { recursive: true })
  await writeFile(join(vaultRoot, "GPU", "Drawings", "diagram.svg"), "<svg>same</svg>")
  await writeFile(join(vaultRoot, "Attachment", "guide.pdf"), "pdf contents")
})

after(async () => {
  await rm(vaultRoot, { recursive: true, force: true })
})

describe("COS asset publishing", () => {
  test("uses content-addressed keys and encoded public URLs", async () => {
    const publisher = createDeterministicAssetPublisher(
      "zhenwei-site/assets",
      "https://assets.example.com/",
    )
    const first = await publisher.publish(join(vaultRoot, "GPU", "Drawings", "diagram.svg"))
    const second = await publisher.publish(join(vaultRoot, "GPU", "Drawings", "diagram.svg"))

    assert.deepEqual(first, second)
    assert.match(first.key, /^zhenwei-site\/assets\/[a-f0-9]{2}\/[a-f0-9]{64}\.svg$/)
    assert.equal(first.url, `https://assets.example.com/${first.key}`)
  })

  test("rewrites image embeds and PDF links without touching code fences", async () => {
    const index = await createVaultAssetIndex(vaultRoot)
    const publisher = createDeterministicAssetPublisher(
      "zhenwei-site/assets",
      "https://assets.example.com",
    )
    const input = `![[GPU/Drawings/diagram.svg|640x480]]
[[Attachment/guide.pdf|CUDA Guide]]

\`\`\`md
![[GPU/Drawings/diagram.svg]]
\`\`\`
`
    const result = await rewriteMarkdownAssetLinks(input, "GPU/note.md", index, publisher)

    assert.match(result.markdown, /!\[diagram\]\(https:\/\/assets\.example\.com\/.+\.svg\)/)
    assert.match(result.markdown, /\[CUDA Guide\]\(https:\/\/assets\.example\.com\/.+\.pdf\)/)
    assert.match(result.markdown, /```md\n!\[\[GPU\/Drawings\/diagram\.svg\]\]\n```/)
    assert.equal(result.assets.length, 2)
  })

  test("rewrites angle-wrapped Markdown image paths from the vault root", async () => {
    const index = await createVaultAssetIndex(vaultRoot)
    const publisher = createDeterministicAssetPublisher("assets", "https://assets.example.com")
    const result = await rewriteMarkdownAssetLinks(
      "![](<\/GPU/Drawings/diagram.svg>)",
      "GPU/note.md",
      index,
      publisher,
    )

    assert.match(result.markdown, /^!\[diagram\]\(https:\/\/assets\.example\.com\/assets\//)
  })

  test("resolves an already-slugged website asset path back to its vault file", async () => {
    await writeFile(join(vaultRoot, "GPU", "Drawings", "GEMM 优化路径栈.svg"), "<svg />")
    const index = await createVaultAssetIndex(vaultRoot)
    const publisher = createDeterministicAssetPublisher("assets", "https://assets.example.com")
    const result = await rewriteMarkdownAssetLinks(
      "![](/gpu/drawings/gemm-优化路径栈.svg)",
      "GPU/note.md",
      index,
      publisher,
    )

    assert.match(result.markdown, /^!\[GEMM 优化路径栈\]\(https:\/\/assets\.example\.com\//)
  })
})
