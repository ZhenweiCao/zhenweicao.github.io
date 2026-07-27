import { parse, stringify } from "yaml"

const FRONTMATTER_PATTERN = /^---\r?\n([\s\S]*?)\r?\n---(?:\r?\n|$)/

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
    items: Array<{
      date: string
      title: string
      href: string
    }>
  }
  footerNote: string
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

function hrefValue(value: unknown, location: string): string {
  const href = stringValue(value, location)
  if (!href.startsWith("/")) {
    throw new Error(`${location} must be site-root-relative and start with "/"`)
  }
  return href
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
  const match = FRONTMATTER_PATTERN.exec(markdown)
  if (match === null) throw new Error("Homepage document must start with YAML frontmatter")

  const frontmatter = asRecord(parse(match[1]), "frontmatter")
  const homepage = asRecord(frontmatter.homepage, "homepage")
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
  knownKeys(featured, ["title", "description", "items"], "homepage.featured")
  if (!Array.isArray(featured.items)) {
    throw new Error("homepage.featured.items must be a sequence")
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
      items: featured.items.map((value, index) => {
        const location = `homepage.featured.items[${index}]`
        const item = asRecord(value, location)
        knownKeys(item, ["date", "title", "href"], location)
        const date = stringValue(item.date, `${location}.date`)
        if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) {
          throw new Error(`${location}.date must use YYYY-MM-DD`)
        }
        return {
          date,
          title: stringValue(item.title, `${location}.title`),
          href: hrefValue(item.href, `${location}.href`),
        }
      }),
    },
    footerNote: stringValue(homepage.footer_note, "homepage.footer_note"),
  }
}

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;")
}

function displayDate(value: string): string {
  return value.replaceAll("-", ".")
}

export function renderHomepage(homepage: HomepageConfiguration, heroImageUrl?: string): string {
  const frontmatter = stringify({
    title: homepage.metaTitle,
    description: homepage.metaDescription,
  })
  const secondaryAction = homepage.hero.secondaryAction
    ? `\n        <a class="home-button" href="${escapeHtml(homepage.hero.secondaryAction.href)}">${escapeHtml(homepage.hero.secondaryAction.label)}</a>`
    : ""
  const preview = heroImageUrl
    ? `<img class="home-preview-image" src="${escapeHtml(heroImageUrl)}" alt="${escapeHtml(homepage.hero.imageAlt ?? "")}">`
    : `<div class="home-preview-window">
        <span class="home-preview-kicker">${escapeHtml(homepage.hero.previewKicker ?? "")}</span>
        <strong class="home-preview-title">${escapeHtml(homepage.hero.previewTitle ?? "")}</strong>
        <div class="home-preview-lines" aria-hidden="true">
          <i></i><i></i><i></i><i></i>
        </div>
      </div>`

  const topics = homepage.topics.items
    .map(
      (item) => `      <a class="home-topic-card" href="${escapeHtml(item.href)}">
        <span>${escapeHtml(item.kicker)}</span>
        <strong>${escapeHtml(item.title)}</strong>
        <small>${escapeHtml(item.description)}</small>
      </a>`,
    )
    .join("\n")
  const featured = homepage.featured.items
    .map(
      (item) => `      <a class="home-reading-item" href="${escapeHtml(item.href)}">
        <time datetime="${item.date}">${displayDate(item.date)}</time>
        <strong>${escapeHtml(item.title)}</strong>
        <span aria-hidden="true">→</span>
      </a>`,
    )
    .join("\n")

  return `---
${frontmatter}---

<!-- Generated from the configured Obsidian homepage document. Do not edit this file directly. -->

<div class="home-shell">
  <section class="home-hero" aria-labelledby="home-title">
    <div>
      <p class="home-badge">${escapeHtml(homepage.hero.badge)}</p>
      <h1 id="home-title">${escapeHtml(homepage.hero.title)}</h1>
      <p class="home-lead">${escapeHtml(homepage.hero.description)}</p>
      <div class="home-actions">
        <a class="home-button is-primary" href="${escapeHtml(homepage.hero.primaryAction.href)}">${escapeHtml(homepage.hero.primaryAction.label)} <span aria-hidden="true">→</span></a>${secondaryAction}
      </div>
    </div>
    <div class="home-preview" aria-label="${escapeHtml(homepage.hero.imageAlt ?? "知识库阅读预览")}">
      ${preview}
    </div>
  </section>

  <section class="home-section" aria-labelledby="topics-title">
    <div class="home-section-heading">
      <h2 id="topics-title">${escapeHtml(homepage.topics.title)}</h2>
      <p>${escapeHtml(homepage.topics.description)}</p>
    </div>
    <div class="home-topic-grid">
${topics}
    </div>
  </section>

  <section class="home-section" aria-labelledby="reading-title">
    <div class="home-section-heading">
      <h2 id="reading-title">${escapeHtml(homepage.featured.title)}</h2>
      <p>${escapeHtml(homepage.featured.description)}</p>
    </div>
    <div class="home-reading-list">
${featured}
    </div>
  </section>

  <aside class="home-note">
    ${escapeHtml(homepage.footerNote)}
  </aside>
</div>
`
}
