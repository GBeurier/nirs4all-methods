/* chemometrics4all — sidebar autoscroll for the corralled toctree-l2 list.
 *
 * When the user is deep inside a section that has many siblings
 * (e.g. one of the 70+ method pages), custom.css clamps the active
 * toctree-l1's child UL to `max-height: 38vh` so it stays scannable.
 * The CSS `scroll-margin-top` alone does not seed the *initial*
 * scroll position; this tiny snippet does.
 *
 * It runs once on DOMContentLoaded, finds the `.current` leaf inside
 * any corralled L1, and scrolls it into the middle of its scrollable
 * parent. Uses `scrollTop` rather than `scrollIntoView({block:
 * 'center'})` so the page itself doesn't also scroll horizontally
 * away from the H1.
 */
(function () {
  'use strict';

  function centerActive() {
    var current = document.querySelector(
      '.sphinxsidebar li.toctree-l1.current > ul li.current'
    );
    if (!current) return;
    var container = current.closest('ul');
    if (!container) return;
    // Element height relative to scroll container, minus half the
    // visible viewport so the active item sits roughly centered.
    var target = current.offsetTop
                  - container.clientHeight / 2
                  + current.clientHeight / 2;
    container.scrollTop = Math.max(0, target);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', centerActive);
  } else {
    centerActive();
  }
})();
