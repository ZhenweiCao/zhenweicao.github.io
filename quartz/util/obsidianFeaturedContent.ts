import { readFile } from "node:fs/promises"
import { dirname, join, posix, resolve } from "node:path"
import { globby } from "globby"
import { parse } from "yaml"
import { slugifyAssetFilePath, slugifyFilePath, type FilePath } from "./path"

const FRONTMATTER_PATTERN = /^---\r?\n([\s\S]*?)\r?\n---(?:\r?\n|$)/

type UnknownRecord = Record<string, unknown>

export type FeaturedArticle = {
  created: string
  updated: string
  title: string
  summary: string
  tags: string[]
  thumbnail?: string
  href: string
  source: string
}

function asRecord(value: unknown, location: string): UnknownRecord {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${location} must be a mapping`)
  }
  return value as UnknownRecord
}

function booleanValue(value: unknown, location: string): boolean {
  if (typeof value !== "boolean") throw new Error(`${location} must be true or false`)
  return value
}

function stringValue(value: unknown, location: string): string {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new Error(`${location} must be a non-empty string`)
  }
  return value.trim()
}

function optionalString(value: unknown, location: string): string | undefined {
  return value === undefined ? undefined : stringValue(value, location)
}

function stringList(value: unknown, location: string): string[] {
  if (value === undefined) return []
  const values = Array.isArray(value) ? value : [value]
  return values.map((item, index) => stringValue(item, `${location}[${index}]`))
}

function dateValue(value: unknown, location: string): string {
  const date = stringValue(value, location)
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date) || Number.isNaN(Date.parse(`${date}T00:00:00Z`))) {
    throw new Error(`${location} must use a valid YYYY-MM-DD date`)
  }
  return date
}

function utcDate(value: Date): Date {
  return new Date(Date.UTC(value.getUTCFullYear(), value.getUTCMonth(), value.getUTCDate()))
}

function subtractCalendarMonths(value: Date, months: number): Date {
  const source = utcDate(value)
  const targetMonthStart = new Date(
    Date.UTC(source.getUTCFullYear(), source.getUTCMonth() - months, 1),
  )
  const lastDay = new Date(
    Date.UTC(targetMonthStart.getUTCFullYear(), targetMonthStart.getUTCMonth() + 1, 0),
  ).getUTCDate()
  targetMonthStart.setUTCDate(Math.min(source.getUTCDate(), lastDay))
  return targetMonthStart
}

function siteHref(relativePath: string): string {
  return `/${slugifyFilePath(relativePath as FilePath)}`
}

function plainText(value: string): string {
  return value
    .replace(/!\[\[[^\]]+\]\]/g, "")
    .replace(/!\[[^\]]*\]\([^)]*\)/g, "")
    .replace(/\[\[([^\]|]+)\|([^\]]+)\]\]/g, "$2")
    .replace(/\[\[([^\]]+)\]\]/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/<[^>]+>/g, "")
    .replace(/[`*~]/g, "")
    .replace(/\s+/g, " ")
    .trim()
}

function truncate(value: string, maxLength = 180): string {
  if (value.length <= maxLength) return value
  const shortened = value
    .slice(0, maxLength)
    .replace(/\s+\S*$/, "")
    .trimEnd()
  return `${shortened || value.slice(0, maxLength)}…`
}

function extractSummary(markdown: string): string {
  const body = markdown.replace(FRONTMATTER_PATTERN, "")
  const candidates: string[] = []

  for (const rawLine of body.split(/\r?\n/)) {
    const line = rawLine.trim()
    if (
      line.length === 0 ||
      /^#{1,6}\s/.test(line) ||
      /^```/.test(line) ||
      /^---+$/.test(line) ||
      /^>\s*\[![^\]]+\]/.test(line)
    ) {
      continue
    }

    const candidate = plainText(
      line
        .replace(/^>\s?/, "")
        .replace(/^[-*+]\s+/, "")
        .replace(/^\d+[.)]\s+/, ""),
    )
    if (candidate.length >= 24) candidates.push(candidate)
    if (candidates.length >= 2) break
  }

  return truncate(candidates.join(" "))
}

function thumbnailHref(reference: string | undefined, source: string): string | undefined {
  if (reference === undefined) return undefined
  if (/^https?:\/\//i.test(reference)) return reference

  const normalized = reference.replaceAll("\\", "/").replace(/^\/+/, "")
  const vaultRelative = normalized.includes("/")
    ? normalized
    : posix.join(dirname(source).replaceAll("\\", "/"), normalized)
  return `/${slugifyAssetFilePath(vaultRelative as FilePath)}`
}

export async function collectRecentFeaturedArticles(options: {
  now?: Date
  vaultRoot: string
  windowMonths: number
}): Promise<FeaturedArticle[]> {
  if (!Number.isInteger(options.windowMonths) || options.windowMonths <= 0) {
    throw new Error("windowMonths must be a positive integer")
  }

  const vaultRoot = resolve(options.vaultRoot)
  const now = utcDate(options.now ?? new Date())
  const cutoff = subtractCalendarMonths(now, options.windowMonths)
  const sourcePaths = await globby("**/*.md", {
    cwd: vaultRoot,
    dot: false,
    onlyFiles: true,
    ignore: ["**/.git/**", "**/node_modules/**", ".obsidian/**"],
  })
  const articles: FeaturedArticle[] = []

  for (const source of sourcePaths.sort()) {
    const markdown = await readFile(join(vaultRoot, source), "utf8")
    const frontmatter = FRONTMATTER_PATTERN.exec(markdown)
    if (frontmatter === null) continue
    if (!/^publish\s*:/m.test(frontmatter[1]) || !/^featured\s*:/m.test(frontmatter[1])) {
      continue
    }
    const metadata = asRecord(parse(frontmatter[1]), `${source} frontmatter`)
    if (metadata.publish !== true || metadata.featured !== true) {
      if (metadata.featured !== undefined && typeof metadata.featured !== "boolean") {
        booleanValue(metadata.featured, `${source} frontmatter.featured`)
      }
      continue
    }

    booleanValue(metadata.publish, `${source} frontmatter.publish`)
    booleanValue(metadata.featured, `${source} frontmatter.featured`)
    const created = dateValue(metadata.created, `${source} frontmatter.created`)
    const updated = dateValue(metadata.updated, `${source} frontmatter.updated`)
    const createdAt = new Date(`${created}T00:00:00Z`)
    if (createdAt < cutoff || createdAt > now) continue

    const configuredSummary = optionalString(
      metadata.description,
      `${source} frontmatter.description`,
    )
    const summary = configuredSummary ?? extractSummary(markdown)
    if (summary.length === 0) {
      throw new Error(
        `${source} needs frontmatter.description or a readable introductory paragraph`,
      )
    }
    const configuredThumbnail =
      optionalString(metadata.cover, `${source} frontmatter.cover`) ??
      optionalString(metadata.thumbnail, `${source} frontmatter.thumbnail`)

    articles.push({
      created,
      updated,
      title: stringValue(metadata.title, `${source} frontmatter.title`),
      summary,
      tags: stringList(metadata.tags, `${source} frontmatter.tags`),
      thumbnail: thumbnailHref(configuredThumbnail, source),
      href: siteHref(source),
      source,
    })
  }

  return articles.sort(
    (left, right) =>
      right.updated.localeCompare(left.updated) ||
      right.created.localeCompare(left.created) ||
      left.title.localeCompare(right.title, "zh-CN") ||
      left.source.localeCompare(right.source),
  )
}
