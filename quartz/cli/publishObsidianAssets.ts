import { readFile, rename, unlink, writeFile } from "node:fs/promises"
import { dirname, join, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { globby } from "globby"
import { createVaultAssetIndex, rewriteMarkdownAssetLinks } from "../util/cosAssetPublisher"
import { createConfiguredAssetPublisher } from "../util/configuredAssetPublisher"
import { loadRepositoryEnvironment } from "../util/localEnvironment"
import {
  OBSIDIAN_SITE_CONFIGURATION_PATH,
  parseObsidianSiteConfiguration,
} from "../util/obsidianSiteConfig"

type CliOptions = {
  check: boolean
  upload: boolean
  vaultRoot: string
}

const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), "../..")
function usage(): string {
  return `Usage: npm run publish:obsidian-assets -- --vault <vault-path> [--check | --upload]

Options:
  --vault <path>  Obsidian vault root (or set OBSIDIAN_VAULT_ROOT)
  --check         Report content whose asset links need regeneration
  --upload        Upload missing content-addressed objects before rewriting links
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
  vaultRoot ??= process.env.OBSIDIAN_VAULT_ROOT
  if (!vaultRoot) throw new Error("Pass --vault <path> or set OBSIDIAN_VAULT_ROOT")

  return { check, upload, vaultRoot: resolve(vaultRoot) }
}

async function atomicWrite(path: string, contents: string): Promise<void> {
  const temporaryPath = join(dirname(path), `.${Date.now()}-${process.pid}.tmp`)
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

  const websiteConfiguration = await readFile(
    join(options.vaultRoot, OBSIDIAN_SITE_CONFIGURATION_PATH),
    "utf8",
  )
  const siteConfiguration = parseObsidianSiteConfiguration(websiteConfiguration)
  const publisher = createConfiguredAssetPublisher(siteConfiguration, options.upload)

  const contentRoot = join(repositoryRoot, "content")
  const markdownFiles = await globby("**/*.md", { cwd: contentRoot, onlyFiles: true })
  const index = await createVaultAssetIndex(options.vaultRoot)
  const changedFiles: Array<{ output: string; targetPath: string }> = []
  const publishedKeys = new Set<string>()

  for (const relativePath of markdownFiles) {
    if (relativePath.toLowerCase() === "index.md") continue
    const targetPath = join(contentRoot, relativePath)
    const currentMarkdown = await readFile(targetPath, "utf8")
    const result = await rewriteMarkdownAssetLinks(currentMarkdown, relativePath, index, publisher)
    result.assets.forEach(({ key }) => publishedKeys.add(key))

    if (currentMarkdown !== result.markdown) {
      changedFiles.push({ output: result.markdown, targetPath })
    }
  }

  if (options.check && changedFiles.length > 0) {
    throw new Error(`${changedFiles.length} published Markdown file(s) have stale asset links`)
  }

  if (!options.check) {
    await Promise.all(changedFiles.map(({ output, targetPath }) => atomicWrite(targetPath, output)))
  }

  console.log(
    `${options.check ? "Checked" : "Processed"} ${markdownFiles.length - 1} Markdown files; ` +
      `${publishedKeys.size} content-addressed assets; ${changedFiles.length} changed files.`,
  )
}

loadRepositoryEnvironment(repositoryRoot)
main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : error)
  console.error(usage())
  process.exitCode = 1
})
