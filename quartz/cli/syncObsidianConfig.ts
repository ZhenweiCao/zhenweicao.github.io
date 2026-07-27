import { readFile, rename, unlink, writeFile } from "node:fs/promises"
import { dirname, join, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { parseObsidianHomepage, renderHomepage } from "../util/obsidianHomepage"
import {
  parseObsidianSiteConfiguration,
  syncSiteConfigurationToQuartz,
} from "../util/obsidianSiteConfig"

type CliOptions = {
  check: boolean
  vaultRoot: string
}

const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), "../..")
const publishingConfigPath = join("VaultMeta", "Publishing", "README.md")

function usage(): string {
  return `Usage: npm run sync:obsidian-config -- --vault <vault-path> [--check]

Options:
  --vault <path>  Obsidian vault root (or set OBSIDIAN_VAULT_ROOT)
  --check         Check for configuration drift without writing
  --help          Show this message`
}

function parseArguments(argv: string[]): CliOptions | null {
  let check = false
  let vaultRoot: string | undefined

  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index]
    if (argument === "--help") return null
    if (argument === "--check") {
      check = true
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

  return { check, vaultRoot: resolve(vaultRoot) }
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

  const sourcePath = join(options.vaultRoot, publishingConfigPath)
  const quartzConfigPath = join(repositoryRoot, "quartz.config.yaml")
  const [publishingReadme, quartzConfig] = await Promise.all([
    readFile(sourcePath, "utf8"),
    readFile(quartzConfigPath, "utf8"),
  ])

  const siteConfiguration = parseObsidianSiteConfiguration(publishingReadme)
  const siteResult = syncSiteConfigurationToQuartz(quartzConfig, siteConfiguration)
  const homepageSourcePath = join(options.vaultRoot, siteConfiguration.homepage.source)
  const homepage = parseObsidianHomepage(await readFile(homepageSourcePath, "utf8"))
  if (homepage.hero.image !== undefined) {
    throw new Error(
      "Homepage local images require the Tencent COS asset publishing feature",
    )
  }

  const homepageOutput = renderHomepage(homepage)
  const homepageTargetPath = join(repositoryRoot, "content", "index.md")
  const currentHomepage = await readFile(homepageTargetPath, "utf8")
  const homepageChanged = currentHomepage !== homepageOutput

  if (!siteResult.changed && !homepageChanged) {
    console.log("Website configuration is in sync with Obsidian.")
    return
  }

  if (options.check) {
    throw new Error(
      `Website configuration differs from ${publishingConfigPath}. Run the sync command without --check.`,
    )
  }

  if (siteResult.changed) await atomicWrite(quartzConfigPath, siteResult.output)
  if (homepageChanged) await atomicWrite(homepageTargetPath, homepageOutput)
  console.log(`Synced website configuration and homepage from ${options.vaultRoot}`)
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : error)
  console.error(usage())
  process.exitCode = 1
})
