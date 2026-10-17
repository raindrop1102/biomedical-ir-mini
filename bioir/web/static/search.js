// Highlight, in the "jump to article" list, the result nearest the top reading line.
// Copied from the original BiomedicalIR search page.
(() => {
  // Track the article nearest the top reading line without changing URL history.
  const articleLinks = [...document.querySelectorAll('.result-navigation a')];
  const articles = articleLinks.map(link => document.getElementById(link.hash.slice(1)));
  let currentArticle = null;
  let jumpTarget = null;
  let jumpTimer;
  function finishJump() {
    clearTimeout(jumpTimer);
    jumpTarget = null;
  }
  function markArticle(article) {
    if (currentArticle === article) return;
    currentArticle = article;
    articleLinks.forEach((link, index) => {
      const active = articles[index] === article;
      if (active) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    });
  }
  function updateReadingPosition() {
    // Native smooth scrolling passes other articles; keep the clicked item selected.
    if (jumpTarget) return;
    const line = Math.min(160, window.innerHeight * 0.25);
    let selected = null;
    for (const article of articles) {
      const rect = article.getBoundingClientRect();
      if (rect.top <= line && rect.bottom > 0) selected = article;
      else if (!selected && rect.top > line && rect.top < window.innerHeight) {
        selected = article;
        break;
      }
    }
    markArticle(selected);
  }
  if (articles.length) {
    let scheduled = false;
    function scheduleReadingUpdate() {
      if (scheduled) return;
      scheduled = true;
      requestAnimationFrame(() => { scheduled = false; updateReadingPosition(); });
    }
    articleLinks.forEach((link, index) => {
      link.addEventListener('click', event => {
        if (event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
        clearTimeout(jumpTimer);
        jumpTarget = articles[index];
        markArticle(jumpTarget);
        // Also release when clicking an anchor that is already in position.
        jumpTimer = setTimeout(finishJump, 1200);
      });
    });
    window.addEventListener('scroll', () => {
      if (jumpTarget) {
        // Quiet-period fallback for browsers without scrollend support.
        clearTimeout(jumpTimer);
        jumpTimer = setTimeout(finishJump, 180);
        return;
      }
      scheduleReadingUpdate();
    }, { passive: true });
    window.addEventListener('scrollend', finishJump);
    // Manual scrolling interrupts the jump and resumes normal tracking.
    window.addEventListener('wheel', finishJump, { passive: true });
    window.addEventListener('touchstart', finishJump, { passive: true });
    window.addEventListener('pointerdown', finishJump, { passive: true });
    window.addEventListener('keydown', event => {
      if (['ArrowUp', 'ArrowDown', 'PageUp', 'PageDown', 'Home', 'End', ' '].includes(event.key)) finishJump();
    });
    window.addEventListener('resize', scheduleReadingUpdate);
    window.addEventListener('hashchange', scheduleReadingUpdate);
    window.addEventListener('pageshow', scheduleReadingUpdate);
    document.querySelectorAll('.result-list details').forEach(detail => {
      detail.addEventListener('toggle', scheduleReadingUpdate);
    });
    updateReadingPosition();
  }
})();
