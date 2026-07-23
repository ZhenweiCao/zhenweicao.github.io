import { PageFrame, PageFrameProps } from "./types"
import HeaderConstructor from "../Header"
import SiteHeaderConstructor from "../SiteHeader"

const Header = HeaderConstructor()
const SiteHeader = SiteHeaderConstructor()

/**
 * The default page frame — three-column layout with left sidebar, center
 * content (header + body + afterBody), and right sidebar, followed by a footer.
 *
 * This is the original Quartz layout, extracted from renderPage.tsx.
 */
export const DefaultFrame: PageFrame = {
  name: "default",
  render({
    componentData,
    header,
    beforeBody,
    pageBody: Content,
    afterBody,
    left,
    right,
    footer: Footer,
  }: PageFrameProps) {
    return (
      <>
        <SiteHeader {...componentData} />
        <aside id="quartz-left-sidebar" class="left sidebar" aria-label="知识导航">
          {left.length > 0 && (
            <button
              class="left-sidebar-toggle"
              type="button"
              aria-controls="quartz-left-sidebar"
              aria-expanded="true"
              aria-label="收起左侧目录层级"
              title="收起左侧目录层级"
            >
              <svg
                class="left-sidebar-icon left-sidebar-icon-collapse"
                xmlns="http://www.w3.org/2000/svg"
                width="18"
                height="18"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
                aria-hidden="true"
                focusable="false"
              >
                <rect width="18" height="18" x="3" y="3" rx="2"></rect>
                <path d="M9 3v18"></path>
                <path d="m16 15-3-3 3-3"></path>
              </svg>
              <svg
                class="left-sidebar-icon left-sidebar-icon-expand"
                xmlns="http://www.w3.org/2000/svg"
                width="18"
                height="18"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
                aria-hidden="true"
                focusable="false"
              >
                <rect width="18" height="18" x="3" y="3" rx="2"></rect>
                <path d="M9 3v18"></path>
                <path d="m14 9 3 3-3 3"></path>
              </svg>
            </button>
          )}
          {left.map((BodyComponent) => (
            <BodyComponent {...componentData} />
          ))}
        </aside>
        <main id="quartz-main" class="center" tabindex={-1}>
          <div class="page-header">
            <Header {...componentData}>
              {header.map((HeaderComponent) => (
                <HeaderComponent {...componentData} />
              ))}
            </Header>
            <div class="popover-hint">
              {beforeBody.map((BodyComponent) => (
                <BodyComponent {...componentData} />
              ))}
            </div>
          </div>
          <Content {...componentData} />
          <hr />
          <div class="page-footer">
            {afterBody.map((BodyComponent) => (
              <BodyComponent {...componentData} />
            ))}
          </div>
        </main>
        <aside class="right sidebar" aria-label="页面辅助信息">
          {right.map((BodyComponent) => (
            <BodyComponent {...componentData} />
          ))}
        </aside>
        <Footer {...componentData} />
      </>
    )
  },
}
