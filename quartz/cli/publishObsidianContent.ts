import { readFile } from "node:fs/promises"
import { dirname, join, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { createConfiguredAssetPublisher } from "../util/configuredAssetPublisher"
import { loadRepositoryEnvironment } from "../util/localEnvironment"
import { syncPublishedObsidianContent } from "../util/obsidianPublishedContent"
import { parseObsidianSiteConfiguration } from "../util/obsidianSiteConfig"

type CliOptions = {
  check: boolean
  upload: boolean
  vaultRoot: string
}

const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), "../..")
const publishingConfigPath = join("VaultMeta", "Publishing", "README.md")

function usage(): string {
  return `Usage: npm run publish:obsidian-content -- --vault <vault-path> [--check | --upload]

Options:
  --vault <path>  Obsidian vault root (or set OBSIDIAN_VAULT_ROOT)
  --check         Check generated content without uploading or writing
  --upload        Upload missing attachments before updating website copies
  --help          Show this message`
}

function parseArguments(argv: string[]): CliOptions | null {
  let check = false
  let upload = false
  let vaultRoot: string | undefined

  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index]
    if (argument === "--help") return null
    if (argument === "--check") {
      check = true
      continue
    }
    if (argument === "--upload") {
      upload = true
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

  if (check && upload) throw new Error("--check and --upload cannot be used together")
  if (!check && !upload) throw new Error("Choose either --check or --upload")
  vaultRoot ??= process.env.OBSIDIAN_VAULT_ROOT
  if (!vaultRoot) throw new Error("Pass --vault <path> or set OBSIDIAN_VAULT_ROOT")

  return { check, upload, vaultRoot: resolve(vaultRoot) }
}

async function main(): Promise<void> {
  const options = parseArguments(process.argv.slice(2))
  if (options === null) {
    console.log(usage())
    return
  }

  const publishingReadme = await readFile(join(options.vaultRoot, publishingConfigPath), "utf8")
  const siteConfiguration = parseObsidianSiteConfiguration(publishingReadme)
  const publisher = createConfiguredAssetPublisher(siteConfiguration, options.upload)
  const result = await syncPublishedObsidianContent({
    check: options.check,
    contentRoot: join(repositoryRoot, "content"),
    excludedSources: [publishingConfigPath, siteConfiguration.homepage.source],
    manifestPath: join(repositoryRoot, ".obsidian-publish-manifest.json"),
    publisher,
    vaultRoot: options.vaultRoot,
  })

  console.log(
    `${options.check ? "Checked" : "Synchronized"} ${result.publishedFiles.length} ` +
      `publish:true note(s); ${result.assetCount} content-addressed asset(s); ` +
      `${result.changedFiles.length} changed file(s); ${result.removedFiles.length} removed file(s).`,
  )
}

loadRepositoryEnvironment(repositoryRoot)
main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : error)
  console.error(usage())
  process.exitCode = 1
})
