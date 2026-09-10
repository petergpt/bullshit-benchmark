# Company logo sources

Retrieved 2026-09-05. All assets are local, byte-for-byte source copies, with no runtime CDN requests. `provenance.json` records each exact download URL, byte count and SHA-256.

## LobeHub Icons

19 SVGs come from [LobeHub Icons](https://github.com/lobehub/lobe-icons), pinned to revision [`4aaf4ee1fb2678a7f989ea570f0f6ce14a9abf75`](https://github.com/lobehub/lobe-icons/tree/4aaf4ee1fb2678a7f989ea570f0f6ce14a9abf75). The primary repository identifies these as AI/LLM brand logos and distributes static SVGs. Its MIT license is preserved verbatim in `LICENSE-lobehub.txt`.

| Local file | Pinned source |
| --- | --- |
| `ai21.svg` | [ai21.svg](https://raw.githubusercontent.com/lobehub/lobe-icons/4aaf4ee1fb2678a7f989ea570f0f6ce14a9abf75/packages/static-svg/icons/ai21.svg) |
| `anthropic.svg` | [anthropic.svg](https://raw.githubusercontent.com/lobehub/lobe-icons/4aaf4ee1fb2678a7f989ea570f0f6ce14a9abf75/packages/static-svg/icons/anthropic.svg) |
| `arcee-ai.svg` | [arcee-color.svg](https://raw.githubusercontent.com/lobehub/lobe-icons/4aaf4ee1fb2678a7f989ea570f0f6ce14a9abf75/packages/static-svg/icons/arcee-color.svg) |
| `baidu.svg` | [baidu-color.svg](https://raw.githubusercontent.com/lobehub/lobe-icons/4aaf4ee1fb2678a7f989ea570f0f6ce14a9abf75/packages/static-svg/icons/baidu-color.svg) |
| `bytedance-seed.svg` | [bytedance-color.svg](https://raw.githubusercontent.com/lobehub/lobe-icons/4aaf4ee1fb2678a7f989ea570f0f6ce14a9abf75/packages/static-svg/icons/bytedance-color.svg) |
| `deepseek.svg` | [deepseek-color.svg](https://raw.githubusercontent.com/lobehub/lobe-icons/4aaf4ee1fb2678a7f989ea570f0f6ce14a9abf75/packages/static-svg/icons/deepseek-color.svg) |
| `google.svg` | [google-color.svg](https://raw.githubusercontent.com/lobehub/lobe-icons/4aaf4ee1fb2678a7f989ea570f0f6ce14a9abf75/packages/static-svg/icons/google-color.svg) |
| `meta.svg` | [meta-color.svg](https://raw.githubusercontent.com/lobehub/lobe-icons/4aaf4ee1fb2678a7f989ea570f0f6ce14a9abf75/packages/static-svg/icons/meta-color.svg) |
| `minimax.svg` | [minimax-color.svg](https://raw.githubusercontent.com/lobehub/lobe-icons/4aaf4ee1fb2678a7f989ea570f0f6ce14a9abf75/packages/static-svg/icons/minimax-color.svg) |
| `mistralai.svg` | [mistral-color.svg](https://raw.githubusercontent.com/lobehub/lobe-icons/4aaf4ee1fb2678a7f989ea570f0f6ce14a9abf75/packages/static-svg/icons/mistral-color.svg) |
| `moonshotai.svg` | [moonshot.svg](https://raw.githubusercontent.com/lobehub/lobe-icons/4aaf4ee1fb2678a7f989ea570f0f6ce14a9abf75/packages/static-svg/icons/moonshot.svg) |
| `nvidia.svg` | [nvidia-color.svg](https://raw.githubusercontent.com/lobehub/lobe-icons/4aaf4ee1fb2678a7f989ea570f0f6ce14a9abf75/packages/static-svg/icons/nvidia-color.svg) |
| `openai.svg` | [openai.svg](https://raw.githubusercontent.com/lobehub/lobe-icons/4aaf4ee1fb2678a7f989ea570f0f6ce14a9abf75/packages/static-svg/icons/openai.svg) |
| `poolside.svg` | [poolside-color.svg](https://raw.githubusercontent.com/lobehub/lobe-icons/4aaf4ee1fb2678a7f989ea570f0f6ce14a9abf75/packages/static-svg/icons/poolside-color.svg) |
| `qwen.svg` | [qwen-color.svg](https://raw.githubusercontent.com/lobehub/lobe-icons/4aaf4ee1fb2678a7f989ea570f0f6ce14a9abf75/packages/static-svg/icons/qwen-color.svg) |
| `stepfun.svg` | [stepfun-color.svg](https://raw.githubusercontent.com/lobehub/lobe-icons/4aaf4ee1fb2678a7f989ea570f0f6ce14a9abf75/packages/static-svg/icons/stepfun-color.svg) |
| `tencent.svg` | [tencent-color.svg](https://raw.githubusercontent.com/lobehub/lobe-icons/4aaf4ee1fb2678a7f989ea570f0f6ce14a9abf75/packages/static-svg/icons/tencent-color.svg) |
| `x-ai.svg` | [xai.svg](https://raw.githubusercontent.com/lobehub/lobe-icons/4aaf4ee1fb2678a7f989ea570f0f6ce14a9abf75/packages/static-svg/icons/xai.svg) |
| `z-ai.svg` | [zai.svg](https://raw.githubusercontent.com/lobehub/lobe-icons/4aaf4ee1fb2678a7f989ea570f0f6ce14a9abf75/packages/static-svg/icons/zai.svg) |

## Xiaomi

`xiaomi.svg` is the company mark from [Simple Icons](https://github.com/simple-icons/simple-icons/blob/7f18aaa676087b8240b6f4ff58a6720be282da59/icons/xiaomi.svg), pinned to revision `7f18aaa676087b8240b6f4ff58a6720be282da59`. Its CC0 license is preserved verbatim in `LICENSE-simple-icons.md`.

## Prime Intellect

`prime-intellect.png` is the unmodified [official icon asset](https://www.primeintellect.ai/icons/logo-icon.png?dpl=dpl_FdZWUWWeCFpfyHKAvRE5NC8BrGqe) referenced by [Prime Intellect’s website](https://www.primeintellect.ai/) for the `primeintellect` organization avatar. The observed deployment identifier is retained in its source URL. This official trademark asset is used solely to identify the benchmark provider; it is not covered by either third-party library license and no open-source license grant is claimed.

## Fallback and color policy

The anonymous `stealth` provider has no verified company identity, so `brandLogo("stealth")` returns `null`. Unknown organizations also return `null`; the UI should display a text initial instead of an invented logo. All 21 identified organizations in the current 22-provider datasets have local assets.

SVGs retain upstream geometry and colors. Series colors are separate: `brandColor()` reproduces the original viewer’s `ORG_COLORS` map and its deterministic HSL fallback, expressed as hex. Brand names and marks remain the property of their respective owners; their appearance does not imply endorsement.
