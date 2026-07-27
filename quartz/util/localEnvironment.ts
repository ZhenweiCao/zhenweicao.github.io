import { existsSync } from "node:fs"
import { loadEnvFile } from "node:process"
import { join } from "node:path"

export function loadRepositoryEnvironment(repositoryRoot: string): void {
  const environmentPath = join(repositoryRoot, ".env")
  if (existsSync(environmentPath)) loadEnvFile(environmentPath)
}
