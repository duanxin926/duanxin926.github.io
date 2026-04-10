const hiddenStandalonePageIds = new Set(["about"]);

export const HIDE_BLOG_LISTING_PAGES = true;

export function isHiddenStandalonePage(pageId: string): boolean {
  return hiddenStandalonePageIds.has(pageId);
}
