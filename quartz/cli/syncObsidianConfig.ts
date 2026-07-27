import { readFile, rename, unlink, writeFile } from "node:fs/promises"
import { dirname, join, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { createVaultAssetIndex, resolveVaultAsset } from "../util/cosAssetPublisher"
import { createConfiguredAssetPublisher } from "../util/configuredAssetPublisher"
import { loadRepositoryEnvironment } from "../util/localEnvironment"
import { parseObsidianHomepage, renderHomepage } from "../util/obsidianHomepage"
import {
  OBSIDIAN_SITE_CONFIGURATION_PATH,
  parseObsidianSiteConfiguration,
  syncSiteConfigurationToQuartz,
} from "../util/obsidianSiteConfig"

type CliOptions = {
  check: boolean
  uploadAssets: boolean
  vaultRoot: string
}

const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), "../..")
function usage(): string {
  return `Usage: npm run sync:obsidian-config -- --vault <vault-path> [--check] [--upload-assets]

Options:
  --vault <path>  Obsidian vault root (or set OBSIDIAN_VAULT_ROOT)
  --check         Check for configuration drift without writing
  --upload-assets Upload a configured homepage image to Tencent COS
  --help          Show this message`
}

function parseArguments(argv: string[]): CliOptions | null {
  let check = false
  let uploadAssets = false
  let vaultRoot: string | undefined

  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index]
    if (argument === "--help") return null
    if (argument === "--check") {
      check = true
      continue
    }
    if (argument === "--upload-assets") {
      uploadAssets = true
      continue
    }
    if (argument === "--vault") {
      vaultRoot = argv[index + 1]
      index += 1
      if (!vaultRoot) throw new Error("--vault requires a path")
      continue
    }
    if (argument.startsWith("--vault=")) {
      vaultRoot = argument.slice("--vault=".length)
      if (!vaultRoot) throw new Error("--vault requires a path")
      continue
    }
    throw new Error(`Unknown argument: ${argument}`)
  }

  vaultRoot ??= process.env.OBSIDIAN_VAULT_ROOT
  if (!vaultRoot) {
    throw new Error("Pass --vault <path> or set OBSIDIAN_VAULT_ROOT")
  }

  if (check && uploadAssets) {
    throw new Error("--check and --upload-assets cannot be used together")
  }

  return { check, uploadAssets, vaultRoot: resolve(vaultRoot) }
}

async function atomicWrite(path: string, contents: string): Promise<void> {
  const temporaryPath = join(dirname(path), `.quartz.config.yaml.${process.pid}.tmp`)
  try {
    await writeFile(temporaryPath, contents, "utf8")
    await rename(temporaryPath, path)
  } catch (error) {
    await unlink(temporaryPath).catch(() => undefined)
    throw error
  }
}

async function main(): Promise<void> {
  const options = parseArguments(process.argv.slice(2))
  if (options === null) {
    console.log(usage())
    return
  }

  const sourcePath = join(options.vaultRoot, OBSIDIAN_SITE_CONFIGURATION_PATH)
  const quartzConfigPath = join(repositoryRoot, "quartz.config.yaml")
  const [websiteConfiguration, quartzConfig] = await Promise.all([
    readFile(sourcePath, "utf8"),
    readFile(quartzConfigPath, "utf8"),
  ])

  const siteConfiguration = parseObsidianSiteConfiguration(websiteConfiguration)
  const siteResult = syncSiteConfigurationToQuartz(quartzConfig, siteConfiguration)
  const homepageSourcePath = join(options.vaultRoot, siteConfiguration.homepage.source)
  const homepage = parseObsidianHomepage(await readFile(homepageSourcePath, "utf8"))
  let heroImageUrl: string | undefined

  if (homepage.hero.image !== undefined) {
    const assets = siteConfiguration.assets
    if (assets === undefined) {
      throw new Error("website.assets is required when the homepage uses a local image")
    }
    const publisher = createConfiguredAssetPublisher(siteConfiguration, options.uploadAssets)
    const index = await createVaultAssetIndex(options.vaultRoot)
    const imagePath = resolveVaultAsset(
      index,
      siteConfiguration.homepage.source,
      homepage.hero.image,
    )
    if (imagePath === undefined) {
      throw new Error(`Homepage image is not a supported local asset: ${homepage.hero.image}`)
    }
    heroImageUrl = (await publisher.publish(imagePath)).url
  }

  const homepageOutput = renderHomepage(homepage, heroImageUrl)
  const homepageTargetPath = join(repositoryRoot, "content", "index.md")
  const currentHomepage = await readFile(homepageTargetPath, "utf8")
  const homepageChanged = currentHomepage !== homepageOutput

  if (!siteResult.changed && !homepageChanged) {
    console.log("Website configuration is in sync with Obsidian.")
    return
  }

  if (options.check) {
    throw new Error(
      `Website configuration differs from ${OBSIDIAN_SITE_CONFIGURATION_PATH}. Run the sync command without --check.`,
    )
  }

  if (siteResult.changed) await atomicWrite(quartzConfigPath, siteResult.output)
  if (homepageChanged) await atomicWrite(homepageTargetPath, homepageOutput)
  console.log(`Synced website configuration and homepage from ${options.vaultRoot}`)
}

loadRepositoryEnvironment(repositoryRoot)
main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : error)
  console.error(usage())
  process.exitCode = 1
})
