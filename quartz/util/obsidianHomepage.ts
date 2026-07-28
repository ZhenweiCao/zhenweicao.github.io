import { parse, parseDocument } from "yaml"
import type { FeaturedArticle } from "./obsidianFeaturedContent"

const HOMEPAGE_CONFIGURATION_PATTERN =
  /<!--\s*quartz-homepage-config\s*-->\s*```(?:yaml|yml)\s*\r?\n([\s\S]*?)\r?\n```/g

type UnknownRecord = Record<string, unknown>

export type HomepageAction = {
  label: string
  href: string
}

export type HomepageConfiguration = {
  metaTitle: string
  metaDescription: string
  hero: {
    badge: string
    title: string
    description: string
    primaryAction: HomepageAction
    secondaryAction?: HomepageAction
    image?: string
    imageAlt?: string
    previewKicker?: string
    previewTitle?: string
  }
  topics: {
    title: string
    description: string
    items: Array<{
      kicker: string
      title: string
      description: string
      href: string
    }>
  }
  featured: {
    title: string
    description: string
    image?: string
    imageAlt?: string
    windowMonths: number
    limit: number
    moreAction: HomepageAction
    generatedOn: string
    items: FeaturedArticle[]
  }
  footerNote?: string
}

function asRecord(value: unknown, location: string): UnknownRecord {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${location} must be a mapping`)
  }
  return value as UnknownRecord
}

function knownKeys(record: UnknownRecord, allowed: readonly string[], location: string): void {
  const allowedKeys = new Set(allowed)
  const unknown = Object.keys(record).filter((key) => !allowedKeys.has(key))
  if (unknown.length > 0) {
    throw new Error(`${location} contains unknown key(s): ${unknown.join(", ")}`)
  }
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

function positiveInteger(value: unknown, location: string): number {
  if (!Number.isInteger(value) || (value as number) <= 0) {
    throw new Error(`${location} must be a positive integer`)
  }
  return value as number
}

function dateValue(value: unknown, location: string): string {
  const date = stringValue(value, location)
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date) || Number.isNaN(Date.parse(`${date}T00:00:00Z`))) {
    throw new Error(`${location} must use a valid YYYY-MM-DD date`)
  }
  return date
}

function hrefValue(value: unknown, location: string): string {
  const href = stringValue(value, location)
  if (!href.startsWith("/")) {
    throw new Error(`${location} must be site-root-relative and start with "/"`)
  }
  return href
}

function optionalHref(value: unknown, location: string): string | undefined {
  if (value === undefined) return undefined
  const href = stringValue(value, location)
  if (!href.startsWith("/") && !/^https?:\/\//i.test(href)) {
    throw new Error(`${location} must be site-root-relative or an HTTP(S) URL`)
  }
  return href
}

function stringList(value: unknown, location: string): string[] {
  if (value === undefined) return []
  if (!Array.isArray(value)) throw new Error(`${location} must be a sequence`)
  return value.map((item, index) => stringValue(item, `${location}[${index}]`))
}

function parseAction(value: unknown, location: string): HomepageAction {
  const action = asRecord(value, location)
  knownKeys(action, ["label", "href"], location)
  return {
    label: stringValue(action.label, `${location}.label`),
    href: hrefValue(action.href, `${location}.href`),
  }
}

export function parseObsidianHomepage(markdown: string): HomepageConfiguration {
  const configurationBlocks = [...markdown.matchAll(HOMEPAGE_CONFIGURATION_PATTERN)]
  if (configurationBlocks.length !== 1) {
    throw new Error(
      "Homepage document must contain exactly one YAML block marked with <!-- quartz-homepage-config -->",
    )
  }

  const configuration = asRecord(parse(configurationBlocks[0][1]), "homepage configuration")
  knownKeys(configuration, ["homepage"], "homepage configuration")
  const homepage = asRecord(configuration.homepage, "homepage")
  knownKeys(
    homepage,
    ["meta_title", "meta_description", "hero", "topics", "featured", "footer_note"],
    "homepage",
  )

  const hero = asRecord(homepage.hero, "homepage.hero")
  knownKeys(
    hero,
    [
      "badge",
      "title",
      "description",
      "primary_action",
      "secondary_action",
      "image",
      "image_alt",
      "preview_kicker",
      "preview_title",
    ],
    "homepage.hero",
  )
  const image = optionalString(hero.image, "homepage.hero.image")
  const imageAlt = optionalString(hero.image_alt, "homepage.hero.image_alt")
  const previewKicker = optionalString(hero.preview_kicker, "homepage.hero.preview_kicker")
  const previewTitle = optionalString(hero.preview_title, "homepage.hero.preview_title")
  if (image === undefined && (previewKicker === undefined || previewTitle === undefined)) {
    throw new Error("homepage.hero needs either image or both preview_kicker and preview_title")
  }

  const topics = asRecord(homepage.topics, "homepage.topics")
  knownKeys(topics, ["title", "description", "items"], "homepage.topics")
  if (!Array.isArray(topics.items)) throw new Error("homepage.topics.items must be a sequence")

  const featured = asRecord(homepage.featured, "homepage.featured")
  knownKeys(
    featured,
    [
      "title",
      "description",
      "image",
      "image_alt",
      "window_months",
      "limit",
      "more_action",
      "results",
    ],
    "homepage.featured",
  )
  const featuredImage = optionalHref(featured.image, "homepage.featured.image")
  const featuredImageAlt = optionalString(featured.image_alt, "homepage.featured.image_alt")
  const featuredResults = asRecord(featured.results, "homepage.featured.results")
  knownKeys(featuredResults, ["generated_on", "items"], "homepage.featured.results")
  if (!Array.isArray(featuredResults.items)) {
    throw new Error("homepage.featured.results.items must be a sequence")
  }

  return {
    metaTitle: stringValue(homepage.meta_title, "homepage.meta_title"),
    metaDescription: stringValue(homepage.meta_description, "homepage.meta_description"),
    hero: {
      badge: stringValue(hero.badge, "homepage.hero.badge"),
      title: stringValue(hero.title, "homepage.hero.title"),
      description: stringValue(hero.description, "homepage.hero.description"),
      primaryAction: parseAction(hero.primary_action, "homepage.hero.primary_action"),
      ...(hero.secondary_action === undefined
        ? {}
        : {
            secondaryAction: parseAction(hero.secondary_action, "homepage.hero.secondary_action"),
          }),
      ...(image === undefined ? {} : { image }),
      ...(imageAlt === undefined ? {} : { imageAlt }),
      ...(previewKicker === undefined ? {} : { previewKicker }),
      ...(previewTitle === undefined ? {} : { previewTitle }),
    },
    topics: {
      title: stringValue(topics.title, "homepage.topics.title"),
      description: stringValue(topics.description, "homepage.topics.description"),
      items: topics.items.map((value, index) => {
        const location = `homepage.topics.items[${index}]`
        const item = asRecord(value, location)
        knownKeys(item, ["kicker", "title", "description", "href"], location)
        return {
          kicker: stringValue(item.kicker, `${location}.kicker`),
          title: stringValue(item.title, `${location}.title`),
          description: stringValue(item.description, `${location}.description`),
          href: hrefValue(item.href, `${location}.href`),
        }
      }),
    },
    featured: {
      title: stringValue(featured.title, "homepage.featured.title"),
      description: stringValue(featured.description, "homepage.featured.description"),
      ...(featuredImage === undefined ? {} : { image: featuredImage }),
      ...(featuredImageAlt === undefined ? {} : { imageAlt: featuredImageAlt }),
      windowMonths: positiveInteger(featured.window_months, "homepage.featured.window_months"),
      limit: positiveInteger(featured.limit, "homepage.featured.limit"),
      moreAction: parseAction(featured.more_action, "homepage.featured.more_action"),
      generatedOn: dateValue(
        featuredResults.generated_on,
        "homepage.featured.results.generated_on",
      ),
      items: featuredResults.items.map((value, index) => {
        const location = `homepage.featured.results.items[${index}]`
        const item = asRecord(value, location)
        knownKeys(
          item,
          ["created", "updated", "title", "summary", "tags", "thumbnail", "href", "source"],
          location,
        )
        return {
          created: dateValue(item.created, `${location}.created`),
          updated: dateValue(item.updated, `${location}.updated`),
          title: stringValue(item.title, `${location}.title`),
          summary: optionalString(item.summary, `${location}.summary`) ?? "",
          tags: stringList(item.tags, `${location}.tags`),
          thumbnail: optionalHref(item.thumbnail, `${location}.thumbnail`),
          href: hrefValue(item.href, `${location}.href`),
          source: stringValue(item.source, `${location}.source`),
        }
      }),
    },
    footerNote: optionalString(homepage.footer_note, "homepage.footer_note"),
  }
}

export function updateHomepageFeaturedResults(
  markdown: string,
  generatedOn: string,
  items: FeaturedArticle[],
): { changed: boolean; output: string } {
  const configurationBlocks = [...markdown.matchAll(HOMEPAGE_CONFIGURATION_PATTERN)]
  if (configurationBlocks.length !== 1) {
    throw new Error(
      "Homepage document must contain exactly one YAML block marked with <!-- quartz-homepage-config -->",
    )
  }

  const block = configurationBlocks[0]
  const document = parseDocument(block[1])
  if (document.errors.length > 0) {
    throw new Error(`Invalid homepage configuration: ${document.errors[0].message}`)
  }
  document.setIn(["homepage", "featured", "results"], {
    generated_on: generatedOn,
    items: items.map(({ created, updated, title, summary, tags, thumbnail, href, source }) => ({
      created,
      updated,
      title,
      summary,
      tags,
      ...(thumbnail === undefined ? {} : { thumbnail }),
      href,
      source,
    })),
  })
  const replacement = `<!-- quartz-homepage-config -->\n\`\`\`yaml\n${document.toString({ lineWidth: 0 }).trimEnd()}\n\`\`\``
  const index = block.index
  if (index === undefined) throw new Error("Homepage configuration block has no source position")
  const output = `${markdown.slice(0, index)}${replacement}${markdown.slice(index + block[0].length)}`
  return { changed: output !== markdown, output }
}

export { renderFeaturedArchive, renderHomepage } from "./obsidianHomepageRender"
