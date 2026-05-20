/* nirs4all-methods — convert sphinx-design tab-sets to a <select> dropdown.
 *
 * sphinx-design emits each tab-set as a flex container of
 * <input type="radio"> + <label> + <div class="sd-tab-content"> triples.
 * Show/hide of content panes is pure CSS via the `:checked + label +
 * .sd-tab-content` adjacent-sibling chain, which keeps working even
 * when labels are visually hidden — so we can hide them and drive the
 * inputs from a single <select>.
 *
 * Behaviour:
 *   - For every .sd-tab-set with ≥ 2 tabs, insert a <div> wrapping a
 *     <select> + a small colored language dot. Options mirror the
 *     visible tab label text and ordering.
 *   - Selecting an option clicks the matching radio input, which lets
 *     sphinx-design's own `:sync` propagation still kick in.
 *   - The dot's color is updated from the option's data-lang attribute.
 *   - Single-tab sets are left untouched.
 */
(function () {
  'use strict';

  function langFromLabel(label) {
    for (const cls of label.classList) {
      if (cls.startsWith('lang-')) return cls.substring(5);
    }
    return 'ext';
  }

  function buildCombo(tabset) {
    const labels = Array.from(
      tabset.querySelectorAll(':scope > .sd-tab-label')
    );
    if (labels.length < 2) return;

    // Wrapper with dot + select.
    const wrap = document.createElement('div');
    wrap.className = 'p4-tab-combo';

    const dot = document.createElement('span');
    dot.className = 'p4-tab-combo-dot';
    wrap.appendChild(dot);

    const select = document.createElement('select');
    select.setAttribute('aria-label', 'Choose a binding');

    let initialLang = 'ext';
    labels.forEach((label) => {
      const opt = document.createElement('option');
      opt.value = label.htmlFor; // id of the matching radio
      opt.textContent = label.textContent.trim();
      const lang = langFromLabel(label);
      opt.dataset.lang = lang;
      const input = document.getElementById(label.htmlFor);
      if (input && input.checked) {
        opt.selected = true;
        initialLang = lang;
      }
      select.appendChild(opt);
    });
    dot.dataset.lang = initialLang;

    select.addEventListener('change', (e) => {
      const selOpt = e.target.options[e.target.selectedIndex];
      const targetId = selOpt.value;
      const input = document.getElementById(targetId);
      if (input) {
        input.checked = true;
        // Fire change so sphinx-design's :sync handler propagates.
        input.dispatchEvent(new Event('change', { bubbles: true }));
        input.dispatchEvent(new Event('click', { bubbles: true }));
      }
      dot.dataset.lang = selOpt.dataset.lang || 'ext';
    });

    wrap.appendChild(select);

    // sphinx-design uses :checked + .sd-tab-label + .sd-tab-content
    // sibling chains, so we mustn't reorder DOM. Insert the combo at
    // the very start of the tab-set — labels stay hidden via CSS,
    // but only on tab-sets we have actually enhanced (the
    // `p4-combo-ready` marker lets CSS opt-in scope label hiding so
    // single-tab sets and no-JS fallback keep working).
    tabset.insertBefore(wrap, tabset.firstChild);
    tabset.classList.add('p4-combo-ready');

    // Keep the combo in sync when the user (or :sync code) toggles
    // a radio input directly — e.g. a synced tab in another tab-set
    // on the same page.
    labels.forEach((label) => {
      const input = document.getElementById(label.htmlFor);
      if (!input) return;
      input.addEventListener('change', () => {
        if (input.checked && select.value !== input.id) {
          select.value = input.id;
          const opt = select.options[select.selectedIndex];
          dot.dataset.lang = opt.dataset.lang || 'ext';
        }
      });
    });
  }

  function init() {
    document.querySelectorAll('.sd-tab-set').forEach(buildCombo);
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
