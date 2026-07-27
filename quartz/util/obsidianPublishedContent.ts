import { access, mkdir, readFile, rename, unlink, writeFile } from "node:fs/promises"
import { dirname, join, relative, resolve } from "node:path"
import { globby } from "globby"
import { parse } from "yaml"
import {
  createVaultAssetIndex,
  rewriteMarkdownAssetLinks,
  type AssetPublisher,
} from "./cosAssetPublisher"

const FRONTMATTER_PATTERN = /^---\r?\n([\s\S]*?)\r?\n---(?:\r?\n|$)/
const MANIFEST_VERSION = 1

type PublishManifest = {
  files: Record<string, string>
  version: typeof MANIFEST_VERSION
}

export type PublishedContentSyncResult = {
  assetCount: number
  changedFiles: string[]
  publishedFiles: string[]
  removedFiles: string[]
}

function normalizeRelativePath(value: string): string {
  return value.replaceAll("\\", "/").replace(/^\/+/, "")
}

function assertSafeRelativePath(value: string, location: string): string {
  const normalized = normalizeRelativePath(value)
  if (!normalized || normalized.split("/").includes("..")) {
    throw new Error(`${location} must be a safe relative path`)
  }
  return normalized
}

function isInside(root: string, candidate: string): boolean {
  const fromRoot = relative(root, candidate)
  return fromRoot !== ".." && !fromRoot.startsWith(`..${process.platform === "win32" ? "\\" : "/"}`)
}

async function exists(path: string): Promise<boolean> {
  try {
    await access(path)
    return true
  } catch {
    return false
  }
}

async function atomicWrite(path: string, contents: string): Promise<void> {
  await mkdir(dirname(path), { recursive: true })
  const temporaryPath = join(dirname(path), `.${Date.now()}-${process.pid}.tmp`)
  try {
    await writeFile(temporaryPath, contents, "utf8")
    await rename(temporaryPath, path)
  } catch (error) {
    await unlink(temporaryPath).catch(() => undefined)
    throw error
  }
}

export function hasPublishAuthorization(markdown: string, sourcePath = "Markdown"): boolean {
  const frontmatter = FRONTMATTER_PATTERN.exec(markdown)
  if (frontmatter === null) return false
  if (!/^publish\s*:/m.test(frontmatter[1])) return false

  const metadata = parse(frontmatter[1]) as unknown
  if (metadata === null || typeof metadata !== "object" || Array.isArray(metadata)) {
    throw new Error(`${sourcePath} frontmatter must be a mapping`)
  }

  const publish = (metadata as Record<string, unknown>).publish
  if (publish === undefined) return false
  if (typeof publish !== "boolean") {
    throw new Error(`${sourcePath} frontmatter publish must be true or false`)
  }
  return publish
}

function parseManifest(contents: string, manifestPath: string): PublishManifest {
  const value = JSON.parse(contents) as unknown
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${manifestPath} must contain an object`)
  }

  const manifest = value as { files?: unknown; version?: unknown }
  if (manifest.version !== MANIFEST_VERSION) {
    throw new Error(`${manifestPath} has an unsupported version`)
  }
  if (
    manifest.files === null ||
    typeof manifest.files !== "object" ||
    Array.isArray(manifest.files)
  ) {
    throw new Error(`${manifestPath}.files must be an object`)
  }

  const files = Object.fromEntries(
    Object.entries(manifest.files as Record<string, unknown>).map(([source, target]) => {
      if (typeof target !== "string") {
        throw new Error(`${manifestPath}.files.${source} must be a string`)
      }
      return [
        assertSafeRelativePath(source, `${manifestPath}.files source`),
        assertSafeRelativePath(target, `${manifestPath}.files.${source}`),
      ]
    }),
  )
  return { files, version: MANIFEST_VERSION }
}

async function readManifest(manifestPath: string): Promise<PublishManifest> {
  if (!(await exists(manifestPath))) return { files: {}, version: MANIFEST_VERSION }
  return parseManifest(await readFile(manifestPath, "utf8"), manifestPath)
}

export async function syncPublishedObsidianContent(options: {
  check: boolean
  contentRoot: string
  excludedSources?: string[]
  manifestPath: string
  publisher: AssetPublisher
  vaultRoot: string
}): Promise<PublishedContentSyncResult> {
  const vaultRoot = resolve(options.vaultRoot)
  const contentRoot = resolve(options.contentRoot)
  const manifestPath = resolve(options.manifestPath)
  const excluded = new Set(
    (options.excludedSources ?? []).map((path) => normalizeRelativePath(path).toLowerCase()),
  )
  const sourcePaths = await globby("**/*.md", {
    cwd: vaultRoot,
    dot: false,
    onlyFiles: true,
    ignore: ["**/.git/**", "**/node_modules/**", ".obsidian/**"],
  })
  const publishedSources: Array<{ markdown: string; relativePath: string }> = []

  for (const relativePath of sourcePaths.sort()) {
    const normalized = normalizeRelativePath(relativePath)
    if (excluded.has(normalized.toLowerCase())) continue
    const markdown = await readFile(join(vaultRoot, normalized), "utf8")
    if (hasPublishAuthorization(markdown, normalized)) {
      publishedSources.push({ markdown, relativePath: normalized })
    }
  }

  const assetIndex = await createVaultAssetIndex(vaultRoot)
  const publishedKeys = new Set<string>()
  const changedFiles: string[] = []
  const nextFiles: Record<string, string> = {}
  const pendingWrites: Array<{ contents: string; targetPath: string }> = []

  for (const source of publishedSources) {
    const targetRelativePath = source.relativePath
    const targetPath = resolve(contentRoot, targetRelativePath)
    if (!isInside(contentRoot, targetPath)) {
      throw new Error(`Published target escapes content root: ${targetRelativePath}`)
    }

    const rewritten = await rewriteMarkdownAssetLinks(
      source.markdown,
      source.relativePath,
      assetIndex,
      options.publisher,
    )
    rewritten.assets.forEach(({ key }) => publishedKeys.add(key))
    const current = (await exists(targetPath)) ? await readFile(targetPath, "utf8") : undefined
    if (current !== rewritten.markdown) {
      changedFiles.push(targetRelativePath)
      pendingWrites.push({ contents: rewritten.markdown, targetPath })
    }
    nextFiles[source.relativePath] = targetRelativePath
  }

  const previousManifest = await readManifest(manifestPath)
  const removedFiles = Object.entries(previousManifest.files)
    .filter(([source]) => nextFiles[source] === undefined)
    .map(([, target]) => target)
    .sort()

  if (options.check && (changedFiles.length > 0 || removedFiles.length > 0)) {
    throw new Error(
      `${changedFiles.length} published file(s) need updating; ${removedFiles.length} managed file(s) need removal`,
    )
  }

  if (!options.check) {
    await Promise.all(
      removedFiles.map(async (target) => {
        const targetPath = resolve(contentRoot, assertSafeRelativePath(target, "manifest target"))
        if (!isInside(contentRoot, targetPath)) {
          throw new Error(`Managed target escapes content root: ${target}`)
        }
        await unlink(targetPath).catch((error: NodeJS.ErrnoException) => {
          if (error.code !== "ENOENT") throw error
        })
      }),
    )
    await Promise.all(
      pendingWrites.map(({ contents, targetPath }) => atomicWrite(targetPath, contents)),
    )
    const manifest: PublishManifest = {
      files: Object.fromEntries(
        Object.entries(nextFiles).sort(([left], [right]) => left.localeCompare(right)),
      ),
      version: MANIFEST_VERSION,
    }
    await atomicWrite(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`)
  }

  return {
    assetCount: publishedKeys.size,
    changedFiles,
    publishedFiles: publishedSources.map(({ relativePath }) => relativePath),
    removedFiles,
  }
}
