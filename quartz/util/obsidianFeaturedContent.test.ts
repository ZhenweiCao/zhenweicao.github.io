import assert from "node:assert/strict"
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { afterEach, describe, test } from "node:test"
import { collectRecentFeaturedArticles } from "./obsidianFeaturedContent"

const temporaryRoots: string[] = []

async function temporaryVault(): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), "quartz-featured-"))
  temporaryRoots.push(root)
  await mkdir(join(root, "Notes"), { recursive: true })
  return root
}

function note(properties: string, body = ""): string {
  return `---\n${properties}---\n# Note\n\n${body}`
}

afterEach(async () => {
  await Promise.all(
    temporaryRoots.splice(0).map((root) => rm(root, { force: true, recursive: true })),
  )
})

describe("Obsidian featured content", () => {
  test("filters recent public featured notes and sorts them by updated date", async () => {
    const vaultRoot = await temporaryVault()
    await Promise.all([
      writeFile(
        join(vaultRoot, "Notes", "Recent.md"),
        note(
          "title: Recent\ncreated: 2026-07-01\nupdated: 2026-07-20\npublish: true\nfeatured: true\ntags:\n  - gpu-kernel\n",
          "> A readable summary extracted from the note introduction.\n\n![[diagram.png]]\n",
        ),
      ),
      writeFile(
        join(vaultRoot, "Notes", "Newer.md"),
        note(
          "title: Newer\ndescription: A configured article summary.\ncover: cover.png\ncreated: 2026-06-01\nupdated: 2026-07-25\npublish: true\nfeatured: true\n",
        ),
      ),
      writeFile(
        join(vaultRoot, "Notes", "Old.md"),
        note(
          "title: Old\ncreated: 2026-01-27\nupdated: 2026-07-27\npublish: true\nfeatured: true\n",
        ),
      ),
      writeFile(
        join(vaultRoot, "Notes", "Private.md"),
        note(
          "title: Private\ncreated: 2026-07-01\nupdated: 2026-07-28\npublish: false\nfeatured: true\n",
        ),
      ),
      writeFile(
        join(vaultRoot, "Notes", "Ordinary.md"),
        note("title: Ordinary\ncreated: 2026-07-01\nupdated: 2026-07-28\npublish: true\n"),
      ),
    ])

    assert.deepEqual(
      await collectRecentFeaturedArticles({
        now: new Date("2026-07-28T12:00:00Z"),
        vaultRoot,
        windowMonths: 6,
      }),
      [
        {
          created: "2026-06-01",
          updated: "2026-07-25",
          title: "Newer",
          summary: "A configured article summary.",
          tags: [],
          thumbnail: "/notes/cover.png",
          href: "/notes/newer",
          source: "Notes/Newer.md",
        },
        {
          created: "2026-07-01",
          updated: "2026-07-20",
          title: "Recent",
          summary: "A readable summary extracted from the note introduction.",
          tags: ["gpu-kernel"],
          thumbnail: undefined,
          href: "/notes/recent",
          source: "Notes/Recent.md",
        },
      ],
    )
  })

  test("requires created and updated dates for a selected note", async () => {
    const vaultRoot = await temporaryVault()
    await writeFile(
      join(vaultRoot, "Notes", "Invalid.md"),
      note("title: Invalid\nupdated: 2026-07-20\npublish: true\nfeatured: true\n"),
    )

    await assert.rejects(
      collectRecentFeaturedArticles({
        now: new Date("2026-07-28T12:00:00Z"),
        vaultRoot,
        windowMonths: 6,
      }),
      /frontmatter.created/,
    )
  })
})
