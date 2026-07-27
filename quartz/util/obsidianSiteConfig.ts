import { isDeepStrictEqual } from "node:util"
import { parse, parseDocument } from "yaml"
import type {
  NavigationConfiguration,
  NavigationLink,
  PinnedNavigationItem,
  SiteHeaderConfiguration,
} from "../cfg"

const SITE_CONFIGURATION_PATTERN =
  /<!--\s*quartz-site-config\s*-->\s*```(?:yaml|yml)\s*\r?\n([\s\S]*?)\r?\n```/g

export const OBSIDIAN_SITE_CONFIGURATION_PATH = "VaultMeta/Publishing/Website.md"

type UnknownRecord = Record<string, unknown>

function asRecord(value: unknown, location: string): UnknownRecord {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${location} must be a mapping`)
  }

  return value as UnknownRecord
}

function assertKnownKeys(record: UnknownRecord, keys: readonly string[], location: string): void {
  const knownKeys = new Set(keys)
  const unknownKeys = Object.keys(record).filter((key) => !knownKeys.has(key))
  if (unknownKeys.length > 0) {
    throw new Error(`${location} contains unknown key(s): ${unknownKeys.join(", ")}`)
  }
}

function requiredString(value: unknown, location: string): string {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new Error(`${location} must be a non-empty string`)
  }

  return value.trim()
}

function optionalString(value: unknown, location: string): string | undefined {
  if (value === undefined) return undefined
  return requiredString(value, location)
}

function validateDirectory(value: unknown, location: string): string | undefined {
  const directory = optionalString(value, location)
  if (directory === undefined) return undefined

  const normalized = directory.replaceAll("\\", "/")
  if (normalized.startsWith("/") || normalized.split("/").includes("..")) {
    throw new Error(`${location} must be relative to the Obsidian vault root`)
  }

  return normalized.replace(/\/+$/, "")
}

function validateHref(value: unknown, location: string): string | undefined {
  const href = optionalString(value, location)
  if (href !== undefined && !href.startsWith("/")) {
    throw new Error(`${location} must be site-root-relative and start with "/"`)
  }

  return href
}

function validateExternalUrl(value: unknown, location: string): string | undefined {
  const url = optionalString(value, location)
  if (url === undefined) return undefined

  let parsed: URL
  try {
    parsed = new URL(url)
  } catch {
    throw new Error(`${location} must be a valid HTTPS URL`)
  }
  if (parsed.protocol !== "https:") {
    throw new Error(`${location} must be a valid HTTPS URL`)
  }

  return url
}

function parseNavigationLink(value: unknown, location: string): NavigationLink {
  const item = asRecord(value, location)
  assertKnownKeys(item, ["label", "directory", "href"], location)

  const label = requiredString(item.label, `${location}.label`)
  const directory = validateDirectory(item.directory, `${location}.directory`)
  const href = validateHref(item.href, `${location}.href`)

  if (directory === undefined && href === undefined) {
    throw new Error(`${location} must define at least one of directory or href`)
  }

  return {
    label,
    ...(directory === undefined ? {} : { directory }),
    ...(href === undefined ? {} : { href }),
  }
}

function parseNavigationItem(value: unknown, index: number): PinnedNavigationItem {
  const location = `website.navigation.items[${index}]`
  const item = asRecord(value, location)
  assertKnownKeys(item, ["label", "directory", "href", "items"], location)

  const label = requiredString(item.label, `${location}.label`)
  const directory = validateDirectory(item.directory, `${location}.directory`)
  const href = validateHref(item.href, `${location}.href`)
  let items: NavigationLink[] = []
  if (item.items !== undefined) {
    if (!Array.isArray(item.items)) throw new Error(`${location}.items must be a sequence`)
    items = item.items.map((child, childIndex) =>
      parseNavigationLink(child, `${location}.items[${childIndex}]`),
    )
  }

  if (directory === undefined && href === undefined && items.length === 0) {
    throw new Error(`${location} must define at least one of directory, href, or items`)
  }

  return {
    label,
    ...(directory === undefined ? {} : { directory }),
    ...(href === undefined ? {} : { href }),
    ...(items.length === 0 ? {} : { items }),
  }
}

export type ObsidianSiteConfiguration = {
  baseUrl: string
  title: string
  siteHeader: SiteHeaderConfiguration
  homepage: {
    source: string
  }
  assets?: {
    provider: "tencent_cos"
    objectPrefix: string
    bucket?: string
    region?: string
    publicBaseUrl?: string
  }
  navigation: NavigationConfiguration
}

export function parseObsidianSiteConfiguration(markdown: string): ObsidianSiteConfiguration {
  const configurationBlocks = [...markdown.matchAll(SITE_CONFIGURATION_PATTERN)]
  if (configurationBlocks.length !== 1) {
    throw new Error(
      "Website configuration document must contain exactly one YAML block marked with <!-- quartz-site-config -->",
    )
  }

  const configuration = asRecord(parse(configurationBlocks[0][1]), "site configuration")
  assertKnownKeys(configuration, ["website"], "site configuration")
  const website = asRecord(configuration.website, "website")
  assertKnownKeys(website, ["base_url", "branding", "homepage", "assets", "navigation"], "website")

  const baseUrl = requiredString(website.base_url, "website.base_url")
  if (
    /^https?:\/\//i.test(baseUrl) ||
    baseUrl.startsWith("/") ||
    baseUrl.endsWith("/") ||
    /[?#\s]/.test(baseUrl)
  ) {
    throw new Error(
      "website.base_url must not include a protocol, whitespace, query, fragment, or surrounding slash",
    )
  }

  const branding = asRecord(website.branding, "website.branding")
  assertKnownKeys(
    branding,
    ["title", "mark", "home_label", "github_url", "github_label"],
    "website.branding",
  )
  const githubUrl = validateExternalUrl(branding.github_url, "website.branding.github_url")
  const githubLabel = optionalString(branding.github_label, "website.branding.github_label")
  if ((githubUrl === undefined) !== (githubLabel === undefined)) {
    throw new Error(
      "website.branding.github_url and website.branding.github_label must be configured together",
    )
  }

  const navigation = asRecord(website.navigation, "website.navigation")
  assertKnownKeys(navigation, ["pinned_limit", "items"], "website.navigation")

  const homepage = asRecord(website.homepage, "website.homepage")
  assertKnownKeys(homepage, ["source"], "website.homepage")
  const homepageSource = validateDirectory(
    website.homepage === undefined ? undefined : homepage.source,
    "website.homepage.source",
  )
  if (homepageSource === undefined || !homepageSource.toLowerCase().endsWith(".md")) {
    throw new Error("website.homepage.source must be a vault-relative Markdown path")
  }

  let assets: ObsidianSiteConfiguration["assets"]
  if (website.assets !== undefined) {
    const rawAssets = asRecord(website.assets, "website.assets")
    assertKnownKeys(
      rawAssets,
      ["provider", "object_prefix", "bucket", "region", "public_base_url"],
      "website.assets",
    )
    const provider = requiredString(rawAssets.provider, "website.assets.provider")
    if (provider !== "tencent_cos") {
      throw new Error('website.assets.provider must be "tencent_cos"')
    }
    const publicBaseUrl = optionalString(
      rawAssets.public_base_url,
      "website.assets.public_base_url",
    )
    if (publicBaseUrl !== undefined && !/^https:\/\//i.test(publicBaseUrl)) {
      throw new Error("website.assets.public_base_url must start with https://")
    }
    const objectPrefix = requiredString(rawAssets.object_prefix, "website.assets.object_prefix")
      .replace(/^\/+/, "")
      .replace(/\/+$/, "")
    if (!objectPrefix) throw new Error("website.assets.object_prefix must not be empty")
    assets = {
      provider,
      objectPrefix,
      ...(rawAssets.bucket === undefined
        ? {}
        : { bucket: requiredString(rawAssets.bucket, "website.assets.bucket") }),
      ...(rawAssets.region === undefined
        ? {}
        : { region: requiredString(rawAssets.region, "website.assets.region") }),
      ...(publicBaseUrl === undefined ? {} : { publicBaseUrl: publicBaseUrl.replace(/\/+$/, "") }),
    }
  }

  const pinnedLimit = navigation.pinned_limit
  if (!Number.isInteger(pinnedLimit) || (pinnedLimit as number) < 0) {
    throw new Error("website.navigation.pinned_limit must be a non-negative integer")
  }

  if (!Array.isArray(navigation.items)) {
    throw new Error("website.navigation.items must be a sequence")
  }

  return {
    baseUrl,
    title: requiredString(branding.title, "website.branding.title"),
    siteHeader: {
      mark: requiredString(branding.mark, "website.branding.mark"),
      homeLabel: requiredString(branding.home_label, "website.branding.home_label"),
      ...(githubUrl === undefined ? {} : { githubUrl, githubLabel }),
    },
    homepage: {
      source: homepageSource,
    },
    ...(assets === undefined ? {} : { assets }),
    navigation: {
      pinnedLimit: pinnedLimit as number,
      pinnedItems: navigation.items.map(parseNavigationItem),
    },
  }
}

export type SiteConfigSyncResult = {
  changed: boolean
  output: string
}

export function syncSiteConfigurationToQuartz(
  quartzConfigYaml: string,
  siteConfiguration: ObsidianSiteConfiguration,
): SiteConfigSyncResult {
  const document = parseDocument(quartzConfigYaml)
  if (document.errors.length > 0) {
    throw new Error(`Invalid quartz.config.yaml: ${document.errors[0].message}`)
  }

  const currentConfig = document.toJS() as {
    configuration?: {
      pageTitle?: unknown
      baseUrl?: unknown
      siteHeader?: unknown
      navigation?: unknown
    }
  }
  if (
    currentConfig.configuration?.pageTitle === siteConfiguration.title &&
    currentConfig.configuration?.baseUrl === siteConfiguration.baseUrl &&
    isDeepStrictEqual(currentConfig.configuration.siteHeader, siteConfiguration.siteHeader) &&
    isDeepStrictEqual(currentConfig.configuration.navigation, siteConfiguration.navigation)
  ) {
    return { changed: false, output: quartzConfigYaml }
  }

  document.setIn(["configuration", "pageTitle"], siteConfiguration.title)
  document.setIn(["configuration", "baseUrl"], siteConfiguration.baseUrl)
  document.setIn(["configuration", "siteHeader"], siteConfiguration.siteHeader)
  document.setIn(["configuration", "navigation"], siteConfiguration.navigation)
  return { changed: true, output: document.toString() }
}
