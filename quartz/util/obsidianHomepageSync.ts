import { access, mkdir, readFile, rename, unlink, writeFile } from "node:fs/promises"
import { basename, dirname, join } from "node:path"
import type { AssetPublisher } from "./cosAssetPublisher"
import { createVaultAssetIndex, resolveVaultAsset } from "./cosAssetPublisher"
import { collectRecentFeaturedArticles } from "./obsidianFeaturedContent"
import {
  parseObsidianHomepage,
  renderFeaturedArchive,
  renderHomepage,
  updateHomepageFeaturedResults,
} from "./obsidianHomepage"
import type { ObsidianSiteConfiguration } from "./obsidianSiteConfig"

export type HomepageSyncResult = {
  archiveChanged: boolean
  featuredCount: number
  homepageChanged: boolean
  sourceChanged: boolean
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
  const temporaryPath = join(dirname(path), `.${basename(path)}.${Date.now()}-${process.pid}.tmp`)
  try {
    await writeFile(temporaryPath, contents, "utf8")
    await rename(temporaryPath, path)
  } catch (error) {
    await unlink(temporaryPath).catch(() => undefined)
    throw error
  }
}

function archiveTargetPath(repositoryRoot: string, href: string): string {
  const relative = href.replace(/^\/+/, "").replace(/\/+$/, "")
  if (!relative || relative.split("/").includes("..")) {
    throw new Error("homepage.featured.more_action.href must identify a safe site page")
  }
  return join(repositoryRoot, "content", `${relative}.md`)
}

function currentDate(now: Date): string {
  return now.toISOString().slice(0, 10)
}

function directImageUrl(value: string): string | undefined {
  return value.startsWith("/") || /^https?:\/\//i.test(value) ? value : undefined
}

export async function syncObsidianHomepage(options: {
  check: boolean
  now?: Date
  publisher: () => AssetPublisher
  repositoryRoot: string
  siteConfiguration: ObsidianSiteConfiguration
  vaultRoot: string
}): Promise<HomepageSyncResult> {
  const now = options.now ?? new Date()
  const homepageSourcePath = join(options.vaultRoot, options.siteConfiguration.homepage.source)
  const currentSource = await readFile(homepageSourcePath, "utf8")
  const initialHomepage = parseObsidianHomepage(currentSource)
  const featuredArticles = await collectRecentFeaturedArticles({
    now,
    vaultRoot: options.vaultRoot,
    windowMonths: initialHomepage.featured.windowMonths,
  })
  const sourceResult = updateHomepageFeaturedResults(
    currentSource,
    currentDate(now),
    featuredArticles,
  )
  const homepage = parseObsidianHomepage(sourceResult.output)
  let heroImageUrl: string | undefined

  if (homepage.hero.image !== undefined) {
    heroImageUrl = directImageUrl(homepage.hero.image)
    if (heroImageUrl === undefined) {
      if (options.siteConfiguration.assets === undefined) {
        throw new Error("website.assets is required when the homepage uses a vault image")
      }
      const index = await createVaultAssetIndex(options.vaultRoot)
      const imagePath = resolveVaultAsset(
        index,
        options.siteConfiguration.homepage.source,
        homepage.hero.image,
      )
      if (imagePath === undefined) {
        throw new Error(`Homepage image is not a supported vault asset: ${homepage.hero.image}`)
      }
      heroImageUrl = (await options.publisher().publish(imagePath)).url
    }
  }

  const homepageTargetPath = join(options.repositoryRoot, "content", "index.md")
  const archivePath = archiveTargetPath(options.repositoryRoot, homepage.featured.moreAction.href)
  const homepageOutput = renderHomepage(homepage, heroImageUrl)
  const archiveOutput = renderFeaturedArchive(homepage)
  const [currentHomepage, currentArchive] = await Promise.all([
    readFile(homepageTargetPath, "utf8"),
    (await exists(archivePath)) ? readFile(archivePath, "utf8") : Promise.resolve(undefined),
  ])
  const homepageChanged = currentHomepage !== homepageOutput
  const archiveChanged = currentArchive !== archiveOutput

  if (options.check && (sourceResult.changed || homepageChanged || archiveChanged)) {
    throw new Error(
      `Homepage drift: source=${sourceResult.changed}, homepage=${homepageChanged}, featuredArchive=${archiveChanged}`,
    )
  }

  if (!options.check) {
    const writes: Array<Promise<void>> = []
    if (sourceResult.changed) writes.push(atomicWrite(homepageSourcePath, sourceResult.output))
    if (homepageChanged) writes.push(atomicWrite(homepageTargetPath, homepageOutput))
    if (archiveChanged) writes.push(atomicWrite(archivePath, archiveOutput))
    await Promise.all(writes)
  }

  return {
    archiveChanged,
    featuredCount: featuredArticles.length,
    homepageChanged,
    sourceChanged: sourceResult.changed,
  }
}
