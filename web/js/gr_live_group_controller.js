import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

// ─── constants ───────────────────────────────────────────────────────────────
const NODE_TYPE   = "GRLiveGroupController";
const WIDGET_NAME = "groupControls";
const SAVE_URL    = "/gr_lgc/save";
const LOAD_URL    = "/gr_lgc/load";

const COLOR = {
  ENABLED_BG:    "#2a7a3b",
  ENABLED_TEXT:  "#e8ffe8",
  DISABLED_BG:   "#3a3a3a",
  DISABLED_TEXT: "#888",
  ACTIVE_BG:     "#e8a020",
  ACTIVE_TEXT:   "#000",
  PANEL_BG:      "#1a1a1a",
  BORDER:        "#555",
  HEADER_TEXT:   "#ccc",
  DRAG_BG:       "#1e3a5f",
  DRAG_TEXT:     "#aaddff",
  DROP_LINE:     "#4a9eff",
  HANDLE:        "#555",
  GOTO:          "#4a9eff",
  SECTION_BG:    "#2a2040",
  SECTION_TEXT:  "#bb99ff",
  SECTION_BORDER:"#6644aa",
  SEARCH_BG:     "#222",
  SEARCH_TEXT:   "#ddd",
  COUNT_BG:      "#333",
  COUNT_TEXT:    "#aaa",
  EDIT_BG:       "#2a3a4a",
  EDIT_TEXT:     "#7ab",
};

const ROW_H    = 36;
const SEC_H    = 26;   // section header row height
const PAD      = 10;
const HDR_H    = 22;
const SEARCH_H = 28;
const MIN_W    = 300;
const HANDLE_W = 18;
const EDIT_W   = 20;  // width of the E (edit/rename) button
const GOTO_W   = 22;
const COUNT_W  = 32;

// ─── save / load ─────────────────────────────────────────────────────────────

async function saveState(data) {
  try {
    await fetch(SAVE_URL, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(data),
    });
  } catch (e) { console.warn("[GR] save failed:", e); }
}

async function loadState() {
  try {
    const res = await fetch(LOAD_URL);
    return await res.json();
  } catch (e) { console.warn("[GR] load failed:", e); return null; }
}

function buildSaveData(items, enabledGroups) {
  return {
    items: items.map((it) => it.kind === "section"
      ? { kind: "section", title: it.title }
      : { kind: "group",   title: it.title }
    ),
    enabledGroups,
  };
}

// ─── canvas helpers ──────────────────────────────────────────────────────────

function getGroups() { return app.graph._groups ?? []; }

function getNodesInGroup(group) {
  const [gx, gy] = group.pos;
  const [gw, gh] = group.size;
  return (app.graph._nodes ?? []).filter((n) => {
    const [nx, ny] = n.pos;
    return nx >= gx && ny >= gy &&
           nx + (n.size[0] ?? 0) <= gx + gw &&
           ny + (n.size[1] ?? 0) <= gy + gh;
  });
}

function setGroupMuted(group, muted) {
  getNodesInGroup(group).forEach((n) => { n.mode = muted ? 4 : 0; });
  app.graph.setDirtyCanvas(true);
}

function isGroupEnabled(group) {
  const nodes = getNodesInGroup(group);
  if (nodes.length === 0) return true;
  return nodes.filter((n) => n.mode === 0).length >= nodes.length / 2;
}

function groupForNode(nodeId) {
  for (const g of getGroups()) {
    if (getNodesInGroup(g).some((n) => String(n.id) === String(nodeId))) return g;
  }
  return null;
}

function renameGroupOnCanvas(group, newTitle) {
  group.title = newTitle;
  app.graph.setDirtyCanvas(true);
}

// ─── focus / back ─────────────────────────────────────────────────────────────

let savedView = null;

function focusGroup(group) {
  const canvas = app.canvas;
  if (!canvas) return;
  savedView = { scale: canvas.ds.scale, offset: [...canvas.ds.offset] };
  const [gx, gy] = group.pos;
  const [gw, gh] = group.size;
  const el = canvas.canvas;
  const vw = el.width / window.devicePixelRatio;
  const vh = el.height / window.devicePixelRatio;
  const scale = Math.min((vw * 0.8) / gw, (vh * 0.8) / gh, 2.0);
  canvas.ds.scale  = scale;
  canvas.ds.offset = [vw / 2 / scale - (gx + gw / 2), vh / 2 / scale - (gy + gh / 2)];
  canvas.setDirty(true, true);
}

function goBack() {
  if (!savedView) return;
  const canvas = app.canvas;
  if (!canvas) return;
  canvas.ds.scale  = savedView.scale;
  canvas.ds.offset = [...savedView.offset];
  savedView = null;
  canvas.setDirty(true, true);
}

window.addEventListener("keyup", (e) => {
  if (e.altKey && !e.ctrlKey && !e.shiftKey && e.key === "b") {
    e.preventDefault();
    goBack();
  }
});

// ─── sync ─────────────────────────────────────────────────────────────────────
// state.items is an array of:
//   { kind:"group",   title, group, enabled }
//   { kind:"section", title }

function syncItems(state) {
  const fresh       = getGroups();
  const freshTitles = new Set(fresh.map((g) => g.title || "(unnamed)"));

  // Refresh existing group entries; remove ones no longer in graph
  const kept = state.items.filter((it) =>
    it.kind === "section" || freshTitles.has(it.title)
  );
  const keptGroupTitles = new Set(
    kept.filter((it) => it.kind === "group").map((it) => it.title)
  );

  kept.forEach((it) => {
    if (it.kind !== "group") return;
    const live = fresh.find((g) => (g.title || "(unnamed)") === it.title);
    if (live) { it.group = live; it.enabled = isGroupEnabled(live); }
  });

  // Append newly added graph groups at the bottom
  const added = fresh
    .filter((g) => !keptGroupTitles.has(g.title || "(unnamed)"))
    .map((g) => ({ kind: "group", title: g.title || "(unnamed)", group: g, enabled: isGroupEnabled(g) }));

  state.items = [...kept, ...added];
}

function applyLoadedState(state, data) {
  if (!data?.items?.length) return;
  syncItems(state);

  const graphGroupMap = Object.fromEntries(
    getGroups().map((g) => [g.title || "(unnamed)", g])
  );
  const currentGroupMap = Object.fromEntries(
    state.items.filter((it) => it.kind === "group").map((it) => [it.title, it])
  );

  // Rebuild items list from saved order, inserting sections and matching groups
  const rebuilt = [];
  for (const saved of data.items) {
    if (saved.kind === "section") {
      rebuilt.push({ kind: "section", title: saved.title });
    } else {
      const existing = currentGroupMap[saved.title];
      if (existing) rebuilt.push(existing);
    }
  }
  // Append any groups not in saved list
  state.items.filter((it) => it.kind === "group" && !rebuilt.find((r) => r.kind === "group" && r.title === it.title))
    .forEach((it) => rebuilt.push(it));

  state.items = rebuilt;
  state.ordered = true;

  if (Array.isArray(data.enabledGroups)) {
    state.items.forEach((it) => {
      if (it.kind !== "group") return;
      it.enabled = data.enabledGroups.includes(it.title);
      setGroupMuted(it.group, !it.enabled);
    });
  }
  app.graph.setDirtyCanvas(true);
}

// ─── rounded rect ─────────────────────────────────────────────────────────────

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

// ─── inline DOM editors (search + rename) ────────────────────────────────────

function makeSearchInput(node, state, widget, onClose) {
  const el = document.createElement("input");
  el.type        = "text";
  el.placeholder = "filter groups…";
  el.style.cssText = `
    position:absolute; z-index:9999; font:12px monospace;
    background:#222; color:#ddd; border:1px solid #4a9eff;
    border-radius:4px; padding:3px 6px; outline:none; box-sizing:border-box;
  `;
  document.body.appendChild(el);

  function reposition() {
    const canvas = app.canvas?.canvas;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const nx   = node.pos[0] * app.canvas.ds.scale + app.canvas.ds.offset[0] * app.canvas.ds.scale + rect.left;
    const ny   = node.pos[1] * app.canvas.ds.scale + app.canvas.ds.offset[1] * app.canvas.ds.scale + rect.top;
    const nw   = node.size[0] * app.canvas.ds.scale;
    el.style.left   = (nx + PAD) + "px";
    el.style.top    = (ny + 30 + HDR_H) + "px";
    el.style.width  = (nw - PAD * 2) + "px";
    el.style.height = SEARCH_H + "px";
  }

  reposition();
  el.focus();

  el.addEventListener("input", () => {
    state.searchText = el.value.toLowerCase();
    app.graph.setDirtyCanvas(true);
  });

  el.addEventListener("keydown", (e) => {
    if (e.key === "Escape") close();
    e.stopPropagation();
  });

  function close() {
    el.remove();
    state.searchText  = "";
    state.searchOpen  = false;
    app.graph.setDirtyCanvas(true);
    onClose?.();
  }

  el.addEventListener("blur", close);
  return { el, reposition, close };
}

function makeRenameInput(node, state, widget, itemIdx, onDone) {
  const item = state.items[itemIdx];
  const el   = document.createElement("input");
  el.type  = "text";
  el.value = item.title;
  el.style.cssText = `
    position:absolute; z-index:9999; font:12px monospace;
    background:#1a1a2e; color:#fff; border:1px solid #bb99ff;
    border-radius:4px; padding:3px 6px; outline:none; box-sizing:border-box;
  `;
  document.body.appendChild(el);

  function reposition() {
    const canvas = app.canvas?.canvas;
    if (!canvas) return;
    const rect  = canvas.getBoundingClientRect();
    const scale = app.canvas.ds.scale;
    const off   = app.canvas.ds.offset;
    const nx    = node.pos[0] * scale + off[0] * scale + rect.left;
    const ny    = node.pos[1] * scale + off[1] * scale + rect.top;
    const nw    = node.size[0] * scale;

    // Compute row Y — count visible rows up to itemIdx
    let visibleIdx = 0;
    for (let i = 0; i < itemIdx; i++) {
      const it = state.items[i];
      const h  = it.kind === "section" ? SEC_H : ROW_H;
      if (!state.searchText || it.kind === "section" || it.title.toLowerCase().includes(state.searchText))
        visibleIdx += h;
    }
    const rowH  = item.kind === "section" ? SEC_H : ROW_H;
    const rowTop = ny + 30 + HDR_H + SEARCH_H + PAD / 2 + visibleIdx;

    el.style.left   = (nx + PAD + 4 + HANDLE_W + 4) + "px";
    el.style.top    = (rowTop + (rowH - 22) / 2) + "px";
    el.style.width  = (nw - PAD * 2 - HANDLE_W - GOTO_W - COUNT_W - 20) + "px";
    el.style.height = "22px";
  }

  reposition();
  el.select();
  el.focus();

  function commit() {
    const newTitle = el.value.trim();
    if (newTitle && newTitle !== item.title) {
      if (item.kind === "group") renameGroupOnCanvas(item.group, newTitle);
      item.title = newTitle;
      onDone?.();
    }
    el.remove();
    state.renaming = false;
    app.graph.setDirtyCanvas(true);
  }

  el.addEventListener("keydown", (e) => {
    if (e.key === "Enter")  commit();
    if (e.key === "Escape") { el.remove(); state.renaming = false; app.graph.setDirtyCanvas(true); }
    e.stopPropagation();
  });
  el.addEventListener("blur", commit);
  return { el, reposition };
}

// ─── widget factory ──────────────────────────────────────────────────────────

function buildWidget(node) {
  const state = {
    items:       [],   // { kind:"group"|"section", title, group?, enabled? }
    activeTitle: null,
    ordered:     false,
    searchText:  "",
    searchOpen:  false,
    renaming:    false,
    drag: { active: false, fromIdx: -1, overIdx: -1 },
  };

  let searchHandle = null;
  let renameHandle = null;

  // Compatibility: state.groups read by websocket handler
  Object.defineProperty(state, "groups", {
    get() { return state.items.filter((it) => it.kind === "group"); }
  });

  function getVisibleItems() {
    if (!state.searchText) return state.items;
    return state.items.filter((it) =>
      it.kind === "section" || it.title.toLowerCase().includes(state.searchText)
    );
  }

  // Row height for an item
  function itemH(it) { return it.kind === "section" ? SEC_H : ROW_H; }

  // Convert widget-local Y to visible item index
  function posToVisibleIdx(posY) {
    const visible = getVisibleItems();
    let offsetY = PAD + HDR_H + SEARCH_H + PAD / 2;
    for (let i = 0; i < visible.length; i++) {
      offsetY += itemH(visible[i]);
      if (posY < offsetY) return i;
    }
    return visible.length;
  }

  // Convert visible index back to state.items index
  function visibleToStateIdx(visIdx) {
    const visible = getVisibleItems();
    if (visIdx < 0 || visIdx >= visible.length) return -1;
    return state.items.indexOf(visible[visIdx]);
  }

  function save() {
    saveState(buildSaveData(
      state.items,
      state.items.filter((it) => it.kind === "group" && it.enabled).map((it) => it.title)
    ));
  }

  const widget = node.addCustomWidget({
    name: WIDGET_NAME,
    type: "custom",

    computeSize() {
      const visible = getVisibleItems();
      const rowsH   = visible.reduce((s, it) => s + itemH(it), 0) || ROW_H;
      return [Math.max(node.size[0], MIN_W), HDR_H + SEARCH_H + PAD + rowsH + PAD];
    },

    draw(ctx, _node, widget_width, y) {
      syncItems(state);

      const w       = widget_width - PAD * 2;
      const x       = PAD;
      let   cy      = y + PAD;
      const { drag } = state;
      const visible  = getVisibleItems();

      // Panel bg
      const totalH = HDR_H + SEARCH_H + PAD + (visible.reduce((s, it) => s + itemH(it), 0) || ROW_H) + PAD;
      ctx.fillStyle   = COLOR.PANEL_BG;
      ctx.strokeStyle = COLOR.BORDER;
      ctx.lineWidth   = 1;
      roundRect(ctx, x, cy, w, totalH, 6);
      ctx.fill();
      ctx.stroke();

      // ── top bar ──
      ctx.fillStyle    = COLOR.HEADER_TEXT;
      ctx.font         = "bold 11px monospace";
      ctx.textAlign    = "left";
      ctx.textBaseline = "middle";
      ctx.fillText("GROUPS  ⠿ drag · ⊙ goto · Alt+B back", x + 10, cy + HDR_H / 2);

      // "+ Section" button top-right
      ctx.fillStyle = COLOR.SECTION_TEXT;
      ctx.font      = "10px monospace";
      ctx.textAlign = "right";
      ctx.fillText("+ section", x + w - 8, cy + HDR_H / 2);

      cy += HDR_H;
      ctx.strokeStyle = COLOR.BORDER;
      ctx.beginPath(); ctx.moveTo(x, cy); ctx.lineTo(x + w, cy); ctx.stroke();

      // ── search bar ──
      ctx.fillStyle   = COLOR.SEARCH_BG;
      ctx.strokeStyle = state.searchOpen ? "#4a9eff" : COLOR.BORDER;
      roundRect(ctx, x + 4, cy + 4, w - 8, SEARCH_H - 8, 4);
      ctx.fill();
      ctx.stroke();
      ctx.strokeStyle = COLOR.BORDER;

      ctx.fillStyle    = state.searchText ? COLOR.SEARCH_TEXT : "#555";
      ctx.font         = "11px monospace";
      ctx.textAlign    = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(
        state.searchText ? state.searchText : "🔍  filter groups…",
        x + 12, cy + SEARCH_H / 2
      );
      if (state.searchText) {
        ctx.fillStyle = "#888";
        ctx.textAlign = "right";
        ctx.fillText("✕", x + w - 10, cy + SEARCH_H / 2);
      }

      cy += SEARCH_H;
      cy += PAD / 2;

      if (visible.length === 0) {
        ctx.fillStyle    = "#555";
        ctx.font         = "11px sans-serif";
        ctx.textAlign    = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(
          state.searchText ? "no matches" : "No groups found in workflow",
          x + w / 2, cy + ROW_H / 2
        );
        return;
      }

      // ── rows ──
      let rowTop = cy;
      visible.forEach((entry, vi) => {
        const si        = state.items.indexOf(entry);
        const isDragged = drag.active && si === drag.fromIdx;
        const rh        = itemH(entry);

        // drop line above
        if (drag.active && vi === drag.overIdx && si !== drag.fromIdx) {
          ctx.fillStyle = COLOR.DROP_LINE;
          ctx.fillRect(x + 4, rowTop - 2, w - 8, 3);
        }

        if (entry.kind === "section") {
          // ── section header ──
          ctx.fillStyle = COLOR.SECTION_BG;
          roundRect(ctx, x + 4, rowTop + 2, w - 8, rh - 4, 4);
          ctx.fill();
          ctx.strokeStyle = COLOR.SECTION_BORDER;
          ctx.lineWidth   = 1;
          roundRect(ctx, x + 4, rowTop + 2, w - 8, rh - 4, 4);
          ctx.stroke();
          ctx.strokeStyle = COLOR.BORDER;

          // drag handle
          ctx.fillStyle    = isDragged ? COLOR.DROP_LINE : COLOR.HANDLE;
          ctx.font         = "13px monospace";
          ctx.textAlign    = "center";
          ctx.textBaseline = "middle";
          ctx.fillText("⠿", x + 4 + HANDLE_W / 2, rowTop + rh / 2);

          // section label with toggle state
          const secGroups = [];
          for (let i = state.items.indexOf(entry) + 1; i < state.items.length; i++) {
            if (state.items[i].kind === "section") break;
            if (state.items[i].kind === "group") secGroups.push(state.items[i]);
          }
          const secAllOff = secGroups.length > 0 && secGroups.every((g) => !g.enabled);
          ctx.fillStyle = secAllOff ? "#7755aa" : COLOR.SECTION_TEXT;
          ctx.font      = "bold 11px monospace";
          ctx.textAlign = "left";
          ctx.fillText((secAllOff ? "✘ " : "✔ ") + entry.title, x + 4 + HANDLE_W + 4, rowTop + rh / 2);

          // E (edit) button for section
          ctx.fillStyle = COLOR.EDIT_BG;
          roundRect(ctx, x + w - 4 - GOTO_W - 4 - EDIT_W, rowTop + rh / 2 - 9, EDIT_W, 18, 3);
          ctx.fill();
          ctx.fillStyle    = COLOR.EDIT_TEXT;
          ctx.font         = "bold 10px monospace";
          ctx.textAlign    = "center";
          ctx.textBaseline = "middle";
          ctx.fillText("E", x + w - 4 - GOTO_W - 4 - EDIT_W / 2, rowTop + rh / 2);

          // delete section button
          ctx.fillStyle = "#664444";
          ctx.font      = "10px monospace";
          ctx.textAlign = "right";
          ctx.fillText("✕", x + w - 10, rowTop + rh / 2);

        } else {
          // ── group row ──
          const { title, enabled } = entry;
          const isActive = title === state.activeTitle;
          let bg   = isDragged ? COLOR.DRAG_BG  : enabled ? COLOR.ENABLED_BG  : COLOR.DISABLED_BG;
          let text = isDragged ? COLOR.DRAG_TEXT : enabled ? COLOR.ENABLED_TEXT : COLOR.DISABLED_TEXT;
          if (isActive && !isDragged) { bg = COLOR.ACTIVE_BG; text = COLOR.ACTIVE_TEXT; }

          ctx.fillStyle = bg;
          roundRect(ctx, x + 4, rowTop + 2, w - 8, rh - 4, 5);
          ctx.fill();

          if (isActive && !isDragged) {
            ctx.strokeStyle = "#ffcc44";
            ctx.lineWidth   = 2;
            roundRect(ctx, x + 4, rowTop + 2, w - 8, rh - 4, 5);
            ctx.stroke();
            ctx.lineWidth = 1;
            ctx.strokeStyle = COLOR.BORDER;
          }

          // drag handle
          ctx.fillStyle    = isDragged ? COLOR.DROP_LINE : COLOR.HANDLE;
          ctx.font         = "14px monospace";
          ctx.textAlign    = "center";
          ctx.textBaseline = "middle";
          ctx.fillText("⠿", x + 4 + HANDLE_W / 2, rowTop + rh / 2);

          // node count badge
          const count = getNodesInGroup(entry.group).length;
          ctx.fillStyle = COLOR.COUNT_BG;
          roundRect(ctx, x + 4 + HANDLE_W + 2, rowTop + rh / 2 - 8, COUNT_W, 16, 3);
          ctx.fill();
          ctx.fillStyle    = COLOR.COUNT_TEXT;
          ctx.font         = "10px monospace";
          ctx.textAlign    = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(count, x + 4 + HANDLE_W + 2 + COUNT_W / 2, rowTop + rh / 2);

          // E (edit) button
          ctx.fillStyle = COLOR.EDIT_BG;
          roundRect(ctx, x + 4 + HANDLE_W + COUNT_W + 6, rowTop + rh / 2 - 9, EDIT_W, 18, 3);
          ctx.fill();
          ctx.fillStyle    = COLOR.EDIT_TEXT;
          ctx.font         = "bold 10px monospace";
          ctx.textAlign    = "center";
          ctx.textBaseline = "middle";
          ctx.fillText("E", x + 4 + HANDLE_W + COUNT_W + 6 + EDIT_W / 2, rowTop + rh / 2);

          // label
          ctx.fillStyle    = text;
          ctx.font         = isActive ? "bold 12px monospace" : "12px monospace";
          ctx.textAlign    = "left";
          ctx.textBaseline = "middle";
          const icon = isActive ? "▶ " : enabled ? "✔ " : "✘ ";
          ctx.fillText(icon + title, x + 4 + HANDLE_W + COUNT_W + EDIT_W + 12, rowTop + rh / 2);

          // goto icon
          ctx.fillStyle = isDragged ? COLOR.HANDLE : COLOR.GOTO;
          ctx.font      = "14px monospace";
          ctx.textAlign = "center";
          ctx.fillText("⊙", x + w - 4 - GOTO_W / 2, rowTop + rh / 2);

          // status tag
          ctx.fillStyle = text;
          ctx.font      = "10px monospace";
          ctx.textAlign = "right";
          ctx.fillText(
            isActive ? "RUNNING" : enabled ? "ON" : "OFF",
            x + w - 4 - GOTO_W - 4, rowTop + rh / 2
          );
        }

        rowTop += rh;
      });

      // drop line at the end
      if (drag.active && drag.overIdx >= visible.length) {
        ctx.fillStyle = COLOR.DROP_LINE;
        ctx.fillRect(x + 4, rowTop - 2, w - 8, 3);
      }

      // resize node
      const needed = widget.computeSize()[1];
      if (Math.abs(node.size[1] - (needed + 60)) > 4) {
        node.setSize([node.size[0], needed + 60]);
        app.graph.setDirtyCanvas(true);
      }
    },

    mouse(event, pos, _node) {
      const { drag } = state;

      if (event.type === "pointerdown") {
        const w = node.size[0] - PAD * 2;

        // ── "+ section" button in header ──
        const headerY = PAD;
        if (pos[1] >= headerY && pos[1] < headerY + HDR_H) {
          const rowX = pos[0] - PAD;
          if (rowX >= w - 70) {
            state.items.push({ kind: "section", title: "Section" });
            state.ordered = true;
            save();
            app.graph.setDirtyCanvas(true);
            // immediately open rename for the new section
            const newIdx = state.items.length - 1;
            const newName = prompt("Section name:", "Section");
            if (newName && newName.trim()) {
              state.items[state.items.length - 1].title = newName.trim();
              save();
              app.graph.setDirtyCanvas(true);
            }
            return true;
          }
          return false;
        }

        // ── search bar ──
        const searchTop = PAD + HDR_H;
        if (pos[1] >= searchTop && pos[1] < searchTop + SEARCH_H) {
          // clear button
          if (state.searchText && pos[0] >= node.size[0] - PAD - 20) {
            state.searchText = "";
            state.searchOpen = false;
            searchHandle?.close();
            searchHandle = null;
            app.graph.setDirtyCanvas(true);
            return true;
          }
          if (!state.searchOpen) {
            state.searchOpen = true;
            searchHandle = makeSearchInput(node, state, widget, () => {
              state.searchOpen = false;
              searchHandle = null;
            });
          }
          return true;
        }

        // ── rows ──
        const vi  = posToVisibleIdx(pos[1]);
        const si  = visibleToStateIdx(vi);
        if (si < 0) return false;

        const entry = state.items[si];
        const rowX  = pos[0] - PAD - 4;

        // drag handle
        if (rowX >= 0 && rowX <= HANDLE_W) {
          drag.active  = true;
          drag.fromIdx = si;
          drag.overIdx = vi;
          app.graph.setDirtyCanvas(true);
          return true;
        }

        if (entry.kind === "section") {
          // E button zone for section
          const secEditLeft = w - 4 - GOTO_W - 4 - EDIT_W;
          const secDeleteLeft = w - 20;
          if (rowX >= secDeleteLeft) {
            state.items.splice(si, 1);
            save();
            app.graph.setDirtyCanvas(true);
            return true;
          }
          if (rowX >= secEditLeft && rowX < secDeleteLeft) {
            const newName = prompt("Rename section:", entry.title);
            if (newName && newName.trim() && newName.trim() !== entry.title) {
              entry.title = newName.trim();
              save();
              app.graph.setDirtyCanvas(true);
            }
            return true;
          }
          // clicking the section body toggles all groups until the next section
          const sectionGroups = [];
          for (let i = si + 1; i < state.items.length; i++) {
            if (state.items[i].kind === "section") break;
            if (state.items[i].kind === "group") sectionGroups.push(state.items[i]);
          }
          if (sectionGroups.length > 0) {
            // if any are enabled, disable all; if all disabled, enable all
            const anyEnabled = sectionGroups.some((g) => g.enabled);
            sectionGroups.forEach((g) => {
              g.enabled = !anyEnabled;
              setGroupMuted(g.group, anyEnabled);
            });
            save();
            app.graph.setDirtyCanvas(true);
          }
          return true;
        }

        // group row
        // goto icon
        const gotoLeft = w - 4 - GOTO_W;
        if (rowX >= gotoLeft) {
          focusGroup(entry.group);
          return true;
        }

        // E button zone
        const editLeft  = HANDLE_W + COUNT_W + 6;
        const editRight = editLeft + EDIT_W;
        if (rowX >= editLeft && rowX <= editRight) {
          const newName = prompt("Rename group:", entry.title);
          if (newName && newName.trim() && newName.trim() !== entry.title) {
            renameGroupOnCanvas(entry.group, newName.trim());
            entry.title = newName.trim();
            syncItems(state);
            save();
            app.graph.setDirtyCanvas(true);
          }
          return true;
        }

        // single click toggle
        entry.enabled = !entry.enabled;
        setGroupMuted(entry.group, !entry.enabled);
        save();
        app.graph.setDirtyCanvas(true);
        return true;
      }

      if (event.type === "pointermove" && drag.active) {
        const vi = posToVisibleIdx(pos[1]);
        drag.overIdx = Math.max(0, Math.min(getVisibleItems().length, vi));
        app.graph.setDirtyCanvas(true);
        return true;
      }

      if (event.type === "pointerup" && drag.active) {
        const { fromIdx, overIdx } = drag;
        drag.active  = false;
        drag.fromIdx = -1;
        drag.overIdx = -1;

        const visible  = getVisibleItems();
        const toStateIdx = (vi) => {
          if (vi >= visible.length) return state.items.length;
          return state.items.indexOf(visible[vi]);
        };

        const insertStateIdx = toStateIdx(overIdx);

        if (insertStateIdx !== fromIdx && insertStateIdx !== fromIdx + 1) {
          const [item]  = state.items.splice(fromIdx, 1);
          const adjustedInsert = insertStateIdx > fromIdx ? insertStateIdx - 1 : insertStateIdx;
          state.items.splice(adjustedInsert, 0, item);
          state.ordered = true;
          save();
        }

        app.graph.setDirtyCanvas(true);
        return true;
      }

      return false;
    },
  });

  node.__lgcState = state;
  return widget;
}

// ─── websocket ───────────────────────────────────────────────────────────────

function getLGCNodes() {
  return (app.graph._nodes ?? []).filter((n) => n.type === NODE_TYPE);
}

api.addEventListener("executing", (evt) => {
  const nodeId = evt.detail;
  for (const node of getLGCNodes()) {
    const state = node.__lgcState;
    if (!state) continue;
    if (nodeId == null) {
      state.activeTitle = null;
    } else {
      const group = groupForNode(nodeId);
      state.activeTitle = group ? (group.title || "(unnamed)") : null;
    }
    app.graph.setDirtyCanvas(true);
  }
});

// ─── register extension ──────────────────────────────────────────────────────

app.registerExtension({
  name: "GRLiveGroupController",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_TYPE) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);
      const node = this;
      buildWidget(node);
      node.setSize([MIN_W + 20, 200]);

      loadState().then((data) => {
        if (!data?.items?.length && !data?.order?.length) return;
        let attempts = 0;
        const apply = () => {
          const state = node.__lgcState;
          if (!state) return;
          if (getGroups().length === 0 && attempts++ < 20) { setTimeout(apply, 100); return; }
          // Support legacy format (just order array)
          if (!data.items && data.order) {
            data = {
              items: data.order.map((t) => ({ kind: "group", title: t })),
              enabledGroups: data.enabledGroups,
            };
          }
          applyLoadedState(state, data);
        };
        setTimeout(apply, 100);
      });
    };
  },
});
