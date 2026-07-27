import assert from "node:assert/strict"
import { describe, test } from "node:test"
import type { NavigationConfiguration } from "../cfg"
import { normalizeNavigationDirectory, resolveNavigationItems } from "./navigation"

describe("site navigation", () => {
  test("normalizes Obsidian directories relative to either vault or content root", () => {
    assert.equal(normalizeNavigationDirectory("GPU/GPU Kernel Learning"), "gpu/gpu-kernel-learning")
    assert.equal(
      normalizeNavigationDirectory("content/GPU/GPU-Kernel-Learning/"),
      "gpu/gpu-kernel-learning",
    )
  })

  test("uses the directory landing page when href is omitted", () => {
    const config: NavigationConfiguration = {
      pinnedItems: [{ label: "知识库", directory: "GPU" }],
    }

    assert.deepEqual(resolveNavigationItems(config, "gpu/index"), [
      {
        label: "知识库",
        href: "/gpu/",
        matchPrefix: "gpu",
        isActive: true,
      },
    ])
  })

  test("limits pinned items while preserving configured order", () => {
    const config: NavigationConfiguration = {
      pinnedLimit: 2,
      pinnedItems: [
        { label: "首页", href: "/" },
        { label: "知识库", directory: "GPU", href: "/gpu/gpu-知识库索引" },
        { label: "标签", directory: "tags" },
      ],
    }

    assert.deepEqual(
      resolveNavigationItems(config, "gpu/gpu-知识库索引").map(({ label, href }) => ({
        label,
        href,
      })),
      [
        { label: "首页", href: "/" },
        { label: "知识库", href: "/gpu/gpu-知识库索引" },
      ],
    )
  })

  test("activates the most specific mapped directory", () => {
    const config: NavigationConfiguration = {
      pinnedItems: [
        { label: "知识库", directory: "GPU", href: "/gpu/gpu-知识库索引" },
        {
          label: "系统课程",
          directory: "GPU/GPU-Kernel-Learning",
          href: "/gpu/gpu-kernel-learning/readme",
        },
      ],
    }

    const items = resolveNavigationItems(config, "gpu/gpu-kernel-learning/chapter-1")

    assert.equal(items[0].isActive, false)
    assert.equal(items[1].isActive, true)
  })

  test("supports hiding all pinned items", () => {
    const config: NavigationConfiguration = {
      pinnedLimit: 0,
      pinnedItems: [{ label: "首页", href: "/" }],
    }

    assert.deepEqual(resolveNavigationItems(config, "index"), [])
  })

  test("infers active state from href when no directory is mapped", () => {
    const config: NavigationConfiguration = {
      pinnedItems: [{ label: "标签", href: "/tags/" }],
    }

    assert.equal(resolveNavigationItems(config, "tags/gpu").at(0)?.isActive, true)
  })
})
