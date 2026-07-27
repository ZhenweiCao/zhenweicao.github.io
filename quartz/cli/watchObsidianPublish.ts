import { spawn } from "node:child_process"
import { extname, resolve } from "node:path"
import { dirname } from "node:path"
import { fileURLToPath } from "node:url"
import { watch } from "chokidar"
import { isPublishableAssetPath } from "../util/cosAssetPublisher"
import { loadRepositoryEnvironment } from "../util/localEnvironment"

type CliOptions = {
  debounceMs: number
  vaultRoot: string
}

const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), "../..")

function usage(): string {
  return `Usage: npm run watch:obsidian-publish -- --vault <vault-path> [--debounce <milliseconds>]

Options:
  --vault <path>          Obsidian vault root (or set OBSIDIAN_VAULT_ROOT)
  --debounce <milliseconds> Delay after the last save before publishing (default: 1000)
  --help                  Show this message`
}

function positiveInteger(value: string | undefined, location: string): number {
  const parsed = Number(value)
  if (!Number.isInteger(parsed) || parsed < 100) {
    throw new Error(`${location} must be an integer of at least 100 milliseconds`)
  }
  return parsed
}

function parseArguments(argv: string[]): CliOptions | null {
  let debounceMs = process.env.OBSIDIAN_AUTO_PUBLISH_DEBOUNCE_MS
    ? positiveInteger(
        process.env.OBSIDIAN_AUTO_PUBLISH_DEBOUNCE_MS,
        "OBSIDIAN_AUTO_PUBLISH_DEBOUNCE_MS",
      )
    : 1000
  let vaultRoot: string | undefined

  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index]
    if (argument === "--help") return null
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
    if (argument === "--debounce") {
      debounceMs = positiveInteger(argv[index + 1], "--debounce")
      index += 1
      continue
    }
    if (argument.startsWith("--debounce=")) {
      debounceMs = positiveInteger(argument.slice("--debounce=".length), "--debounce")
      continue
    }
    throw new Error(`Unknown argument: ${argument}`)
  }

  vaultRoot ??= process.env.OBSIDIAN_VAULT_ROOT
  if (!vaultRoot) throw new Error("Pass --vault <path> or set OBSIDIAN_VAULT_ROOT")
  return { debounceMs, vaultRoot: resolve(vaultRoot) }
}

function runNpmScript(script: string, vaultRoot: string, extraArguments: string[]): Promise<void> {
  const executable = process.platform === "win32" ? "npm.cmd" : "npm"
  return new Promise((resolvePromise, reject) => {
    const child = spawn(
      executable,
      ["run", script, "--", "--vault", vaultRoot, ...extraArguments],
      {
        cwd: repositoryRoot,
        env: process.env,
        stdio: "inherit",
      },
    )
    child.once("error", reject)
    child.once("exit", (code, signal) => {
      if (code === 0) {
        resolvePromise()
        return
      }
      reject(
        new Error(
          `${script} failed${signal ? ` with signal ${signal}` : ` with exit code ${code ?? "unknown"}`}`,
        ),
      )
    })
  })
}

async function publish(vaultRoot: string, reasons: string[]): Promise<void> {
  const startedAt = new Date().toLocaleTimeString()
  console.log(`\n[${startedAt}] Publishing after: ${reasons.sort().join(", ")}`)
  await runNpmScript("sync:obsidian-config", vaultRoot, ["--upload-assets"])
  await runNpmScript("publish:obsidian-content", vaultRoot, ["--upload"])
  console.log(`[${new Date().toLocaleTimeString()}] Obsidian publication synchronized.`)
}

function shouldObserve(path: string): boolean {
  return extname(path).toLowerCase() === ".md" || isPublishableAssetPath(path)
}

function isIgnoredPath(path: string): boolean {
  const ignoredSegments = new Set([".git", ".obsidian", ".trash", "node_modules"])
  return path.split(/[\\/]/).some((segment) => ignoredSegments.has(segment))
}

async function main(): Promise<void> {
  const options = parseArguments(process.argv.slice(2))
  if (options === null) {
    console.log(usage())
    return
  }

  let running = false
  let queued = false
  let timer: NodeJS.Timeout | undefined
  const reasons = new Set<string>()

  const runQueuedPublish = async (): Promise<void> => {
    if (running) {
      queued = true
      return
    }
    running = true
    const currentReasons = [...reasons]
    reasons.clear()
    try {
      await publish(options.vaultRoot, currentReasons)
    } catch (error) {
      console.error(
        `[${new Date().toLocaleTimeString()}] Automatic publication failed:`,
        error instanceof Error ? error.message : error,
      )
    } finally {
      running = false
      if (queued || reasons.size > 0) {
        queued = false
        await runQueuedPublish()
      }
    }
  }

  const queuePublish = (reason: string): void => {
    reasons.add(reason)
    if (timer !== undefined) clearTimeout(timer)
    timer = setTimeout(() => void runQueuedPublish(), options.debounceMs)
  }

  const watcher = watch(options.vaultRoot, {
    awaitWriteFinish: {
      pollInterval: 100,
      stabilityThreshold: 500,
    },
    ignoreInitial: true,
    ignored: isIgnoredPath,
  })

  watcher.on("all", (event, path) => {
    if (shouldObserve(path)) queuePublish(`${event}:${path}`)
  })
  watcher.on("error", (error) => console.error("Obsidian watcher error:", error))

  const close = async (): Promise<void> => {
    if (timer !== undefined) clearTimeout(timer)
    await watcher.close()
    process.exitCode = 0
  }
  process.once("SIGINT", () => void close())
  process.once("SIGTERM", () => void close())

  console.log(`Watching ${options.vaultRoot}`)
  await publish(options.vaultRoot, ["watcher-start"])
}

loadRepositoryEnvironment(repositoryRoot)
main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : error)
  console.error(usage())
  process.exitCode = 1
})
