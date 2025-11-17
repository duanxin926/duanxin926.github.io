/**
 * Post Filtering Script for Posts Pages
 * Handles clickable category and tag pills to filter post cards
 */

function initializePostFiltering() {
  const categoryItems = document.querySelectorAll('.category-item[data-category]');
  const tagItems = document.querySelectorAll('.tag-item[data-tag]');
  const postCards = document.querySelectorAll('[data-post-categories], [data-post-tags]');
  
  // Initialize from URL params if present (for backward compatibility)
  const urlParams = new URLSearchParams(window.location.search);
  let currentActiveCategory = urlParams.get('category') || null;
  let currentActiveTag = urlParams.get('tag') || null;
  
  // Also check URL path for category/tag (e.g., /posts/category/xxx or /posts/tag/xxx)
  const pathname = window.location.pathname;
  if (!currentActiveCategory && pathname.includes('/category/')) {
    const match = pathname.match(/\/category\/([^\/]+)/);
    if (match) {
      currentActiveCategory = decodeURIComponent(match[1]);
    }
  }
  if (!currentActiveTag && pathname.includes('/tag/')) {
    const match = pathname.match(/\/tag\/([^\/]+)/);
    if (match) {
      currentActiveTag = decodeURIComponent(match[1]);
    }
  }

  if (categoryItems.length === 0 && tagItems.length === 0) {
    return;
  }

  // Filter function that checks both category and tag
  function filterPosts() {
    postCards.forEach((card) => {
      const cardCategoriesStr = card.getAttribute('data-post-categories') || '';
      const cardTagsStr = card.getAttribute('data-post-tags') || '';
      
      // Split into arrays for exact matching
      const cardCategories = cardCategoriesStr ? cardCategoriesStr.split(',').map(c => c.trim()) : [];
      const cardTags = cardTagsStr ? cardTagsStr.split(',').map(t => t.trim()) : [];
      
      let shouldShow = true;
      
      // Check category filter (exact match)
      if (currentActiveCategory) {
        shouldShow = shouldShow && cardCategories.includes(currentActiveCategory);
      }
      
      // Check tag filter (exact match)
      if (currentActiveTag) {
        shouldShow = shouldShow && cardTags.includes(currentActiveTag);
      }
      
      // Show or hide the card
      if (shouldShow) {
        card.style.display = 'block';
      } else {
        card.style.display = 'none';
      }
    });
  }

  // Update button states - re-query elements to get fresh references
  function updateButtonStates() {
    const freshCategoryItems = document.querySelectorAll('.category-item[data-category]');
    const freshTagItems = document.querySelectorAll('.tag-item[data-tag]');
    
    // Update category buttons
    freshCategoryItems.forEach((btn) => {
      const category = btn.getAttribute('data-category');
      const isActive = currentActiveCategory === category;
      
      btn.classList.remove(
        'bg-highlight-100', 'dark:bg-highlight-900/40', 
        'text-highlight-800', 'dark:text-highlight-200', 
        'ring-1', 'ring-highlight-300', 'dark:ring-highlight-700',
        'bg-primary-100', 'dark:bg-primary-800', 
        'text-primary-700', 'dark:text-primary-300'
      );
      
      if (isActive) {
        btn.classList.add(
          'bg-highlight-100', 'dark:bg-highlight-900/40', 
          'text-highlight-800', 'dark:text-highlight-200', 
          'ring-1', 'ring-highlight-300', 'dark:ring-highlight-700'
        );
      } else {
        btn.classList.add(
          'bg-primary-100', 'dark:bg-primary-800', 
          'text-primary-700', 'dark:text-primary-300'
        );
      }
    });
    
    // Update tag buttons
    freshTagItems.forEach((btn) => {
      const tag = btn.getAttribute('data-tag');
      const isActive = currentActiveTag === tag;
      
      btn.classList.remove(
        'bg-highlight-100', 'dark:bg-highlight-900/40', 
        'text-highlight-800', 'dark:text-highlight-200', 
        'ring-1', 'ring-highlight-300', 'dark:ring-highlight-700',
        'bg-primary-100', 'dark:bg-primary-800', 
        'text-primary-700', 'dark:text-primary-300'
      );
      
      if (isActive) {
        btn.classList.add(
          'bg-highlight-100', 'dark:bg-highlight-900/40', 
          'text-highlight-800', 'dark:text-highlight-200', 
          'ring-1', 'ring-highlight-300', 'dark:ring-highlight-700'
        );
      } else {
        btn.classList.add(
          'bg-primary-100', 'dark:bg-primary-800', 
          'text-primary-700', 'dark:text-primary-300'
        );
      }
    });
  }

  // Remove existing event listeners and add new ones for categories
  categoryItems.forEach((item) => {
    const newItem = item.cloneNode(true);
    item.parentNode?.replaceChild(newItem, item);
    
    newItem.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      
      const category = newItem.getAttribute('data-category');
      
      if (!category) return;
      
      // Toggle category selection
      if (currentActiveCategory === category) {
        currentActiveCategory = null;
      } else {
        currentActiveCategory = category;
      }
      
      // Filter posts and update button states
      filterPosts();
      updateButtonStates();
      
      // Dispatch custom event for analytics
      window.dispatchEvent(new CustomEvent('categoryClicked', {
        detail: { category: currentActiveCategory, isActive: currentActiveCategory !== null }
      }));
    });
  });

  // Remove existing event listeners and add new ones for tags
  tagItems.forEach((item) => {
    const newItem = item.cloneNode(true);
    item.parentNode?.replaceChild(newItem, item);
    
    newItem.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      
      const tag = newItem.getAttribute('data-tag');
      
      if (!tag) return;
      
      // Toggle tag selection
      if (currentActiveTag === tag) {
        currentActiveTag = null;
      } else {
        currentActiveTag = tag;
      }
      
      // Filter posts and update button states
      filterPosts();
      updateButtonStates();
      
      // Dispatch custom event for analytics
      window.dispatchEvent(new CustomEvent('tagClicked', {
        detail: { tag: currentActiveTag, isActive: currentActiveTag !== null }
      }));
    });
  });

  // Initial filter and button state update
  filterPosts();
  updateButtonStates();
}

// Initialize with multiple strategies to ensure it works
function tryInitialize() {
  initializePostFiltering();
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', tryInitialize);

// Re-initialize after Swup page transitions
if (typeof window !== 'undefined' && window.swup) {
  window.swup.hooks.on('content:replace', tryInitialize);
  window.swup.hooks.on('page:view', tryInitialize);
  window.swup.hooks.on('visit:end', tryInitialize);
}

// Also try to initialize immediately in case DOM is already ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', tryInitialize);
} else {
  // Use setTimeout to ensure DOM is fully ready
  setTimeout(tryInitialize, 100);
}

// Fallback: try again after a short delay
setTimeout(tryInitialize, 500);

// Export for manual initialization if needed
if (typeof window !== 'undefined') {
  window.initializePostFiltering = initializePostFiltering;
}

