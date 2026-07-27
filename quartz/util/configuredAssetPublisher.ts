import type { ObsidianSiteConfiguration } from "./obsidianSiteConfig"
import {
  createDeterministicAssetPublisher,
  createTencentCosAssetPublisher,
  type AssetPublisher,
} from "./cosAssetPublisher"

function requiredSetting(value: string | undefined, name: string): string {
  if (!value) throw new Error(`Set ${name} before publishing assets`)
  return value
}

export function createConfiguredAssetPublisher(
  siteConfiguration: ObsidianSiteConfiguration,
  upload: boolean,
): AssetPublisher {
  const assets = siteConfiguration.assets
  if (assets === undefined) throw new Error("website.assets is not configured")

  const publicBaseUrl = requiredSetting(
    assets.publicBaseUrl ?? process.env.TENCENT_COS_PUBLIC_BASE_URL,
    "website.assets.public_base_url or TENCENT_COS_PUBLIC_BASE_URL",
  )
  if (!upload) {
    return createDeterministicAssetPublisher(assets.objectPrefix, publicBaseUrl)
  }

  return createTencentCosAssetPublisher({
    bucket: requiredSetting(
      assets.bucket ?? process.env.TENCENT_COS_BUCKET,
      "website.assets.bucket or TENCENT_COS_BUCKET",
    ),
    region: requiredSetting(
      assets.region ?? process.env.TENCENT_COS_REGION,
      "website.assets.region or TENCENT_COS_REGION",
    ),
    objectPrefix: assets.objectPrefix,
    publicBaseUrl,
    secretId: process.env.TENCENT_COS_SECRET_ID,
    secretKey: process.env.TENCENT_COS_SECRET_KEY,
    securityToken: process.env.TENCENT_COS_SECURITY_TOKEN,
  })
}
