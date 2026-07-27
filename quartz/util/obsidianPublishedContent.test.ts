import assert from "node:assert/strict"
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises"
import { join } from "node:path"
import { tmpdir } from "node:os"
import { afterEach, describe, test } from "node:test"
import { createDeterministicAssetPublisher } from "./cosAssetPublisher"
import { hasPublishAuthorization, syncPublishedObsidianContent } from "./obsidianPublishedContent"

const temporaryRoots: string[] = []

async function temporaryRoot(): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), "quartz-publish-"))
  temporaryRoots.push(root)
  return root
}

afterEach(async () => {
  await Promise.all(
    temporaryRoots.splice(0).map((root) => rm(root, { force: true, recursive: true })),
  )
})

describe("Obsidian published content", () => {
  test("only accepts a boolean publish property in YAML frontmatter", () => {
    assert.equal(hasPublishAuthorization("---\npublish: true\n---\nPublic"), true)
    assert.equal(hasPublishAuthorization("---\npublish: false\n---\nPrivate"), false)
    assert.equal(hasPublishAuthorization("---\ntitle: Note\n---\nDraft"), false)
    assert.equal(hasPublishAuthorization("---\npaper: Invalid: YAML\n---\nDraft"), false)
    assert.equal(hasPublishAuthorization("publish: true"), false)
    assert.throws(
      () => hasPublishAuthorization("---\npublish: yes\n---\nInvalid", "Invalid.md"),
      /publish must be true or false/,
    )
  })

  test("exports authorized notes and rewrites their direct assets", async () => {
    const root = await temporaryRoot()
    const vaultRoot = join(root, "vault")
    const contentRoot = join(root, "content")
    const manifestPath = join(root, "manifest.json")
    await Promise.all([
      mkdir(join(vaultRoot, "Notes"), { recursive: true }),
      mkdir(join(vaultRoot, "Attachment"), { recursive: true }),
      mkdir(contentRoot, { recursive: true }),
    ])
    await Promise.all([
      writeFile(
        join(vaultRoot, "Notes", "Public.md"),
        "---\npublish: true\n---\n# Public\n\n![[Attachment/figure.png]]\n",
      ),
      writeFile(join(vaultRoot, "Notes", "Private.md"), "---\npublish: false\n---\n# Private\n"),
      writeFile(join(vaultRoot, "Attachment", "figure.png"), Buffer.from("image")),
    ])

    const result = await syncPublishedObsidianContent({
      check: false,
      contentRoot,
      manifestPath,
      publisher: createDeterministicAssetPublisher("site/assets", "https://assets.example.com"),
      vaultRoot,
    })

    assert.deepEqual(result.publishedFiles, ["Notes/Public.md"])
    assert.equal(result.assetCount, 1)
    assert.match(
      await readFile(join(contentRoot, "Notes", "Public.md"), "utf8"),
      /https:\/\/assets\.example\.com\/site\/assets\/[a-f0-9]{2}\/[a-f0-9]{64}\.png/,
    )
    await assert.rejects(readFile(join(contentRoot, "Notes", "Private.md"), "utf8"), /ENOENT/)
  })

  test("removes only previously managed files after publish is disabled", async () => {
    const root = await temporaryRoot()
    const vaultRoot = join(root, "vault")
    const contentRoot = join(root, "content")
    const manifestPath = join(root, "manifest.json")
    await Promise.all([
      mkdir(join(vaultRoot, "Notes"), { recursive: true }),
      mkdir(join(contentRoot, "Manual"), { recursive: true }),
    ])
    const sourcePath = join(vaultRoot, "Notes", "Managed.md")
    await Promise.all([
      writeFile(sourcePath, "---\npublish: true\n---\n# Managed\n"),
      writeFile(join(contentRoot, "Manual", "Keep.md"), "# Keep\n"),
    ])
    const options = {
      check: false,
      contentRoot,
      manifestPath,
      publisher: createDeterministicAssetPublisher("site/assets", "https://assets.example.com"),
      vaultRoot,
    }
    await syncPublishedObsidianContent(options)
    await writeFile(sourcePath, "---\npublish: false\n---\n# Managed\n")

    const result = await syncPublishedObsidianContent(options)

    assert.deepEqual(result.removedFiles, ["Notes/Managed.md"])
    await assert.rejects(readFile(join(contentRoot, "Notes", "Managed.md"), "utf8"), /ENOENT/)
    assert.equal(await readFile(join(contentRoot, "Manual", "Keep.md"), "utf8"), "# Keep\n")
  })
})
