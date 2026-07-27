import { createHash } from "node:crypto"
import { readFile } from "node:fs/promises"
import { basename, dirname, extname, relative, resolve } from "node:path"
import COS from "cos-nodejs-sdk-v5"
import { globby } from "globby"
import { slugTag } from "./path"

const PUBLISHABLE_EXTENSIONS = new Set([
  ".avif",
  ".csv",
  ".gif",
  ".jpeg",
  ".jpg",
  ".json",
  ".mp3",
  ".mp4",
  ".pdf",
  ".png",
  ".svg",
  ".txt",
  ".wav",
  ".webm",
  ".webp",
  ".zip",
])

const IMAGE_EXTENSIONS = new Set([".avif", ".gif", ".jpeg", ".jpg", ".png", ".svg", ".webp"])

const CONTENT_TYPES: Record<string, string> = {
  ".avif": "image/avif",
  ".csv": "text/csv; charset=utf-8",
  ".gif": "image/gif",
  ".jpeg": "image/jpeg",
  ".jpg": "image/jpeg",
  ".json": "application/json",
  ".mp3": "audio/mpeg",
  ".mp4": "video/mp4",
  ".pdf": "application/pdf",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".txt": "text/plain; charset=utf-8",
  ".wav": "audio/wav",
  ".webm": "video/webm",
  ".webp": "image/webp",
  ".zip": "application/zip",
}

export type PublishedAsset = {
  key: string
  sourcePath: string
  url: string
}

export type AssetPublisher = {
  publish(sourcePath: string): Promise<PublishedAsset>
}

export type VaultAssetIndex = {
  byBasename: Map<string, string[]>
  byRelativePath: Map<string, string>
  vaultRoot: string
}

export type CosAssetConfiguration = {
  bucket: string
  objectPrefix: string
  publicBaseUrl: string
  region: string
  secretId?: string
  secretKey?: string
  securityToken?: string
}

function normalizeRelativePath(value: string): string {
  return value
    .replaceAll("\\", "/")
    .replace(/^\/+/, "")
    .replace(/^(?:\.\/)+/, "")
}

function isInside(root: string, candidate: string): boolean {
  const pathFromRoot = relative(root, candidate)
  return (
    pathFromRoot !== ".." &&
    !pathFromRoot.startsWith(`..${process.platform === "win32" ? "\\" : "/"}`)
  )
}

function decodeLinkTarget(value: string): string {
  const unwrapped = value.trim().replace(/^<|>$/g, "").split("#", 1)[0]
  try {
    return decodeURIComponent(unwrapped)
  } catch {
    return unwrapped
  }
}

export function isPublishableAssetPath(path: string): boolean {
  return PUBLISHABLE_EXTENSIONS.has(extname(path).toLowerCase())
}

export async function createVaultAssetIndex(vaultRoot: string): Promise<VaultAssetIndex> {
  const absoluteRoot = resolve(vaultRoot)
  const relativePaths = await globby("**/*", {
    cwd: absoluteRoot,
    dot: false,
    onlyFiles: true,
  })
  const byRelativePath = new Map<string, string>()
  const byBasename = new Map<string, string[]>()

  for (const relativePath of relativePaths) {
    if (!isPublishableAssetPath(relativePath)) continue
    const normalized = normalizeRelativePath(relativePath)
    const absolutePath = resolve(absoluteRoot, normalized)
    byRelativePath.set(normalized.toLowerCase(), absolutePath)
    const extension = extname(normalized)
    const sluggedPath = `${slugTag(normalized.slice(0, -extension.length))}${extension.toLowerCase()}`
    byRelativePath.set(sluggedPath, absolutePath)

    const name = basename(normalized).toLowerCase()
    const matches = byBasename.get(name) ?? []
    matches.push(absolutePath)
    byBasename.set(name, matches)
  }

  return { byBasename, byRelativePath, vaultRoot: absoluteRoot }
}

export function resolveVaultAsset(
  index: VaultAssetIndex,
  sourceNoteRelativePath: string,
  rawTarget: string,
): string | undefined {
  const target = normalizeRelativePath(decodeLinkTarget(rawTarget))
  if (!target || /^(?:[a-z]+:)?\/\//i.test(target) || !isPublishableAssetPath(target)) {
    return undefined
  }

  const candidates = [normalizeRelativePath(`${dirname(sourceNoteRelativePath)}/${target}`), target]
  for (const candidate of candidates) {
    const match = index.byRelativePath.get(candidate.toLowerCase())
    if (match !== undefined && isInside(index.vaultRoot, match)) return match
  }

  if (!target.includes("/")) {
    const matches = index.byBasename.get(target.toLowerCase()) ?? []
    if (matches.length === 1) return matches[0]
    if (matches.length > 1) {
      throw new Error(
        `Ambiguous Obsidian attachment "${rawTarget}" from ${sourceNoteRelativePath}; use a vault-relative path`,
      )
    }
  }

  throw new Error(
    `Cannot resolve Obsidian attachment "${rawTarget}" from ${sourceNoteRelativePath}`,
  )
}

function encodeObjectKey(key: string): string {
  return key
    .split("/")
    .map((segment) => encodeURIComponent(segment))
    .join("/")
}

export async function describePublishedAsset(
  sourcePath: string,
  objectPrefix: string,
  publicBaseUrl: string,
): Promise<PublishedAsset> {
  const contents = await readFile(sourcePath)
  const digest = createHash("sha256").update(contents).digest("hex")
  const extension = extname(sourcePath).toLowerCase()
  const normalizedPrefix = objectPrefix.replace(/^\/+|\/+$/g, "")
  const key = `${normalizedPrefix}/${digest.slice(0, 2)}/${digest}${extension}`
  return {
    key,
    sourcePath,
    url: `${publicBaseUrl.replace(/\/+$/, "")}/${encodeObjectKey(key)}`,
  }
}

export function createDeterministicAssetPublisher(
  objectPrefix: string,
  publicBaseUrl: string,
): AssetPublisher {
  if (!objectPrefix.replace(/^\/+|\/+$/g, "")) {
    throw new Error("COS object prefix must not be empty")
  }
  if (!/^https:\/\//i.test(publicBaseUrl)) {
    throw new Error("COS public base URL must start with https://")
  }
  const cache = new Map<string, Promise<PublishedAsset>>()
  return {
    publish(sourcePath) {
      const absolutePath = resolve(sourcePath)
      const cached = cache.get(absolutePath)
      if (cached !== undefined) return cached
      const result = describePublishedAsset(absolutePath, objectPrefix, publicBaseUrl)
      cache.set(absolutePath, result)
      return result
    },
  }
}

function isNotFound(error: unknown): boolean {
  if (error === null || typeof error !== "object") return false
  const candidate = error as { code?: unknown; statusCode?: unknown }
  return candidate.statusCode === 404 || candidate.code === "NoSuchKey"
}

export function createTencentCosAssetPublisher(config: CosAssetConfiguration): AssetPublisher {
  if (!config.bucket || !config.region) {
    throw new Error("Tencent COS bucket and region are required when uploading assets")
  }
  if (!config.secretId || !config.secretKey) {
    throw new Error(
      "TENCENT_COS_SECRET_ID and TENCENT_COS_SECRET_KEY are required when uploading assets",
    )
  }

  const cos = new COS({
    SecretId: config.secretId,
    SecretKey: config.secretKey,
    ...(config.securityToken ? { SecurityToken: config.securityToken } : {}),
  })
  const deterministic = createDeterministicAssetPublisher(config.objectPrefix, config.publicBaseUrl)
  const uploads = new Map<string, Promise<PublishedAsset>>()

  return {
    async publish(sourcePath) {
      const descriptor = await deterministic.publish(sourcePath)
      const existing = uploads.get(descriptor.key)
      if (existing !== undefined) return existing

      const upload = (async () => {
        try {
          await cos.headObject({
            Bucket: config.bucket,
            Region: config.region,
            Key: descriptor.key,
          })
        } catch (error) {
          if (!isNotFound(error)) throw error
          await cos.uploadFile({
            Bucket: config.bucket,
            Region: config.region,
            Key: descriptor.key,
            FilePath: descriptor.sourcePath,
            ContentType: CONTENT_TYPES[extname(descriptor.sourcePath).toLowerCase()],
            CacheControl: "public, max-age=31536000, immutable",
          })
        }
        return descriptor
      })()

      uploads.set(descriptor.key, upload)
      return upload
    },
  }
}

function fencedCodeRanges(markdown: string): Array<[number, number]> {
  const ranges: Array<[number, number]> = []
  const pattern = /^(?<marker>`{3,}|~{3,})[^\n]*\n[\s\S]*?^\k<marker>\s*$/gm
  for (const match of markdown.matchAll(pattern)) {
    ranges.push([match.index, match.index + match[0].length])
  }
  return ranges
}

function isInRanges(offset: number, ranges: Array<[number, number]>): boolean {
  return ranges.some(([start, end]) => offset >= start && offset < end)
}

function wikiLabel(target: string, alias: string | undefined): string {
  if (alias && !/^\d+(?:%?x\d+)?$/.test(alias.trim())) return alias.trim()
  return basename(target, extname(target))
}

type Replacement = {
  end: number
  start: number
  value: string
}

export async function rewriteMarkdownAssetLinks(
  markdown: string,
  sourceNoteRelativePath: string,
  index: VaultAssetIndex,
  publisher: AssetPublisher,
): Promise<{ assets: PublishedAsset[]; markdown: string }> {
  const ranges = fencedCodeRanges(markdown)
  const replacements: Replacement[] = []
  const assets = new Map<string, PublishedAsset>()

  const wikiPattern = /(!?)\[\[([^\]\n]+)\]\]/g
  for (const match of markdown.matchAll(wikiPattern)) {
    if (isInRanges(match.index, ranges)) continue
    const [target, alias] = match[2].split("|", 2)
    const sourcePath = resolveVaultAsset(index, sourceNoteRelativePath, target)
    if (sourcePath === undefined) continue
    const published = await publisher.publish(sourcePath)
    assets.set(published.key, published)
    const label = wikiLabel(target, alias)
    const isImage = IMAGE_EXTENSIONS.has(extname(sourcePath).toLowerCase())
    replacements.push({
      start: match.index,
      end: match.index + match[0].length,
      value:
        match[1] === "!" && isImage
          ? `![${label}](${published.url})`
          : `[${label}](${published.url})`,
    })
  }

  const markdownPattern = /(!?)\[([^\]\n]*)\]\((<[^>\n]+>|[^)\n]+)\)/g
  for (const match of markdown.matchAll(markdownPattern)) {
    if (
      isInRanges(match.index, ranges) ||
      replacements.some(({ start, end }) => match.index >= start && match.index < end)
    ) {
      continue
    }
    const sourcePath = resolveVaultAsset(index, sourceNoteRelativePath, match[3])
    if (sourcePath === undefined) continue
    const published = await publisher.publish(sourcePath)
    assets.set(published.key, published)
    const isImage = IMAGE_EXTENSIONS.has(extname(sourcePath).toLowerCase())
    const label = match[2] || basename(sourcePath, extname(sourcePath))
    replacements.push({
      start: match.index,
      end: match.index + match[0].length,
      value:
        match[1] === "!" && isImage
          ? `![${label}](${published.url})`
          : `[${label}](${published.url})`,
    })
  }

  replacements.sort((left, right) => right.start - left.start)
  let rewritten = markdown
  for (const replacement of replacements) {
    rewritten =
      rewritten.slice(0, replacement.start) + replacement.value + rewritten.slice(replacement.end)
  }

  return { assets: [...assets.values()], markdown: rewritten }
}
