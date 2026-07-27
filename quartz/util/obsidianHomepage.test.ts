import assert from "node:assert/strict"
import { describe, test } from "node:test"
import { parseObsidianHomepage, renderHomepage } from "./obsidianHomepage"

const homepageDocument = `---
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
    items:
      - date: 2026-07-25
        title: First
        href: /first
  footer_note: Footer
---
# Homepage
`

describe("Obsidian homepage", () => {
  test("parses all configurable homepage sections", () => {
    const homepage = parseObsidianHomepage(homepageDocument)

    assert.equal(homepage.hero.image, "Attachment/Image/hero.png")
    assert.equal(homepage.topics.items[0].title, "GPU")
    assert.equal(homepage.featured.items[0].date, "2026-07-25")
  })

  test("renders remote hero images and escapes configured text", () => {
    const output = renderHomepage(
      parseObsidianHomepage(homepageDocument),
      "https://assets.example.com/site/hash.png",
    )

    assert.match(output, /A &lt;safe&gt; title/)
    assert.match(output, /src="https:\/\/assets\.example\.com\/site\/hash\.png"/)
    assert.match(output, /<time datetime="2026-07-25">2026\.07\.25<\/time>/)
    assert.doesNotMatch(output, /Attachment\/Image\/hero\.png/)
  })

  test("requires either an image or preview text", () => {
    const invalid = homepageDocument
      .replace("    image: Attachment/Image/hero.png\n", "")
      .replace("    image_alt: Hero image\n", "")

    assert.throws(() => parseObsidianHomepage(invalid), /needs either image/)
  })
})
