import assert from "node:assert/strict"
import { describe, test } from "node:test"
import {
  parseObsidianHomepage,
  renderFeaturedArchive,
  renderHomepage,
  updateHomepageFeaturedResults,
} from "./obsidianHomepage"

const homepageDocument = `---
title: Homepage
content_type: reference
---
# Homepage

<!-- quartz-homepage-config -->
\`\`\`yaml
homepage:
  meta_title: Test Blog
  meta_description: Test description
  hero:
    badge: Notes
    title: A <safe> title
    description: Description
    primary_action:
      label: Start
      href: /start
    image: Attachment/Image/hero.png
    image_alt: Hero image
  topics:
    title: Topics
    description: Topic description
    items:
      - kicker: 01 · GPU
        title: GPU
        description: GPU description
        href: /gpu/
  featured:
    title: Featured
    description: Featured description
    image: /static/visuals/featured-editorial.avif
    image_alt: Featured visual
    window_months: 6
    limit: 1
    more_action:
      label: View all
      href: /featured/
    results:
      generated_on: 2026-07-28
      items:
        - created: 2026-07-01
          updated: 2026-07-25
          title: First
          summary: First summary
          tags:
            - gpu-kernel
            - performance
          thumbnail: /images/first.png
          href: /first
          source: Notes/First.md
  footer_note: Footer
\`\`\`
`

describe("Obsidian homepage", () => {
  test("parses all configurable homepage sections", () => {
    const homepage = parseObsidianHomepage(homepageDocument)

    assert.equal(homepage.hero.image, "Attachment/Image/hero.png")
    assert.equal(homepage.topics.items[0].title, "GPU")
    assert.equal(homepage.featured.limit, 1)
    assert.equal(homepage.featured.image, "/static/visuals/featured-editorial.avif")
    assert.equal(homepage.featured.items[0].updated, "2026-07-25")
    assert.equal(homepage.featured.items[0].summary, "First summary")
    assert.deepEqual(homepage.featured.items[0].tags, ["gpu-kernel", "performance"])
  })

  test("renders remote hero images and escapes configured text", () => {
    const output = renderHomepage(
      parseObsidianHomepage(homepageDocument),
      "https://assets.example.com/site/hash.png",
    )

    assert.match(output, /A &lt;safe&gt; title/)
    assert.match(output, /src="https:\/\/assets\.example\.com\/site\/hash\.png"/)
    assert.match(output, /<time datetime="2026-07-25">更新于 2026\.07\.25<\/time>/)
    assert.match(output, /href="\/featured\/">View all/)
    assert.match(output, /src="\/images\/first\.png"/)
    assert.match(output, /href="\/tags\/gpu-kernel"/)
    assert.match(output, /src="\/static\/visuals\/featured-editorial\.avif"/)
    assert.ok(output.indexOf("home-section is-topics") < output.indexOf("home-section is-featured"))
    assert.doesNotMatch(output, /Attachment\/Image\/hero\.png/)
  })

  test("omits the homepage note when it is not configured", () => {
    const withoutFooterNote = homepageDocument.replace("  footer_note: Footer\n", "")
    const output = renderHomepage(parseObsidianHomepage(withoutFooterNote))

    assert.doesNotMatch(output, /home-note/)
  })

  test("updates generated featured results in the managed body block", () => {
    const result = updateHomepageFeaturedResults(homepageDocument, "2026-07-29", [
      {
        created: "2026-07-02",
        updated: "2026-07-29",
        title: "Generated",
        summary: "Generated summary",
        tags: ["gpu-kernel"],
        thumbnail: "/images/generated.png",
        href: "/generated",
        source: "Notes/Generated.md",
      },
    ])
    const homepage = parseObsidianHomepage(result.output)

    assert.equal(result.changed, true)
    assert.equal(homepage.featured.generatedOn, "2026-07-29")
    assert.equal(homepage.featured.items[0].source, "Notes/Generated.md")
    assert.equal(homepage.featured.items[0].thumbnail, "/images/generated.png")
    assert.doesNotMatch(result.output.split("---", 3)[1], /featured:/)
  })

  test("renders every generated item on the featured archive", () => {
    const output = renderFeaturedArchive(parseObsidianHomepage(homepageDocument))

    assert.match(output, /unlisted: true/)
    assert.match(output, /featured-card-list is-archive/)
    assert.match(output, /featured-archive-visual/)
    assert.match(output, /href="\/first"/)
  })

  test("requires either an image or preview text", () => {
    const invalid = homepageDocument
      .replace("    image: Attachment/Image/hero.png\n", "")
      .replace("    image_alt: Hero image\n", "")

    assert.throws(() => parseObsidianHomepage(invalid), /needs either image/)
  })

  test("does not accept homepage configuration from document properties", () => {
    const propertiesOnly = `---
title: Homepage
homepage:
  meta_title: Test Blog
---
# Homepage
`

    assert.throws(() => parseObsidianHomepage(propertiesOnly), /exactly one YAML block marked/)
  })

  test("rejects multiple marked homepage configuration blocks", () => {
    const duplicated = `${homepageDocument}
<!-- quartz-homepage-config -->
\`\`\`yaml
homepage: {}
\`\`\`
`

    assert.throws(() => parseObsidianHomepage(duplicated), /exactly one YAML block marked/)
  })
})
