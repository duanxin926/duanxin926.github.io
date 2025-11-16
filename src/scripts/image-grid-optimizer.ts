/**
 * Image Grid Optimizer
 * Automatically classifies images in grids based on their aspect ratio
 * and applies appropriate styling for better visual consistency
 */

function classifyImagesByRatio() {
  const imageGrids = document.querySelectorAll('.prose .image-grid');
  
  imageGrids.forEach(grid => {
    const images = grid.querySelectorAll('img');
    
    images.forEach(img => {
      // Wait for image to load
      if (img.complete && img.naturalWidth > 0) {
        classifyImage(img);
      } else {
        img.addEventListener('load', () => classifyImage(img));
      }
    });
  });
}

function classifyImage(img: HTMLImageElement) {
  const ratio = img.naturalWidth / img.naturalHeight;
  
  // Remove any existing ratio classes
  img.classList.remove('ratio-landscape', 'ratio-portrait', 'ratio-square');
  
  // Classify based on aspect ratio
  if (ratio > 1.3) {
    // Wide/landscape image (e.g., 16:9, 4:3)
    img.classList.add('ratio-landscape');
  } else if (ratio < 0.75) {
    // Tall/portrait image (e.g., 3:4, 9:16)
    img.classList.add('ratio-portrait');
  } else {
    // Square or near-square image (e.g., 1:1, 4:5)
    img.classList.add('ratio-square');
  }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', classifyImagesByRatio);

// Re-initialize after Swup page transitions (if Swup is available)
if (window.swup) {
  window.swup.hooks.on('page:view', classifyImagesByRatio);
  window.swup.hooks.on('visit:end', classifyImagesByRatio);
}

// Export for manual triggering
(window as any).classifyImagesByRatio = classifyImagesByRatio;

