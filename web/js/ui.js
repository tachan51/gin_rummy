import {
  GinMatch, cardName, isRed, rankOf, suitOf, RANK_LABELS, SUIT_SYMBOLS,
  deadwood, discardDeadwoods, arrange, bySuitRank, byRankSuit, KNOCK_LIMIT, BOX_BONUS, GAME_BONUS,
} from './engine.js';
import { decide } from './ai.js';

const HUMAN = 0, AI = 1;
const LEVEL_NAMES = { easy: 'よわい', normal: 'ふつう', strong: '最強' };
// A whole AI turn (draw + discard, thinking included) takes about 0.5 s at normal speed:
// each action waits max(thinking time, AI_WAIT).
const AI_WAIT = 240;

// AI runs in a Web Worker so the page stays responsive; falls back to the main thread.
let worker = null;
let reqId = 0;
const pending = new Map();
try {
  worker = new Worker(new URL('./ai-worker.js', import.meta.url), { type: 'module' });
  worker.onmessage = (e) => { const p = pending.get(e.data.id); if (p) { pending.delete(e.data.id); p(e.data.action); } };
  worker.onerror = () => { worker = null; for (const [, p] of pending) p(null); pending.clear(); };
} catch { worker = null; }
function askAI(view, level) {
  if (worker) {
    return new Promise((resolve) => {
      const id = ++reqId;
      pending.set(id, (a) => resolve(a || decide(view, level)));
      worker.postMessage({ id, view, level });
    });
  }
  return Promise.resolve(decide(view, level));
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const nowMs = () => performance.now();

const $ = (id) => document.getElementById(id);
const store = {
  get(k, d) { try { const v = localStorage.getItem('ginrummy.' + k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem('ginrummy.' + k, JSON.stringify(v)); } catch { /* ignore */ } },
};

const settings = Object.assign({ level: 'strong', target: 100, sort: 'meld', showAI: false, speed: 1 }, store.get('settings', {}));
// level meanings changed in v2 (easy/normal/strong were re-assigned), so records start fresh
let stats = store.get('stats2', {});

let match = null;
let hand = null;
let selected = null;
let freshCard = null;
let hintCard = null;
let hintText = '';
let busy = false;
let message = '';
let token = 0; // invalidates pending AI timers on new game

// ---------------------------------------------------------------------------
// rendering helpers
// ---------------------------------------------------------------------------
function cardEl(c, extra = '') {
  const el = document.createElement('div');
  const r = rankOf(c);
  el.className = `card ${isRed(c) ? 'red' : ''} ${r >= 10 ? 'face' : ''} ${extra}`;
  const rank = RANK_LABELS[r], suit = SUIT_SYMBOLS[suitOf(c)];
  el.innerHTML = `<span class="corner">${rank}<span class="s">${suit}</span></span><span class="pip">${r >= 10 ? rank + suit : suit}</span>`;
  el.setAttribute('aria-label', cardName(c));
  el.dataset.card = c;
  return el;
}
function backEl() { const el = document.createElement('div'); el.className = 'card back'; return el; }

function fitOverlap(container, count, varName) {
  const w = container.clientWidth || 320;
  const cw = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--cw')) || 60;
  const gaps = 30;
  let ov = count > 1 ? (w - gaps - cw * count) / (count - 1) : 0;
  ov = Math.min(6, Math.max(-cw * 0.68, ov));
  container.style.setProperty(varName, ov + 'px');
}

/** groups for the human hand according to the sort mode */
function groupHand(cards) {
  if (settings.sort === 'suit') return [{ cards: cards.slice().sort(bySuitRank), meld: -1 }];
  if (settings.sort === 'rank') return [{ cards: cards.slice().sort(byRankSuit), meld: -1 }];
  const a = arrange(cards);
  const groups = a.melds.map((m, i) => ({ cards: m, meld: i }));
  if (a.deadwood.length) groups.push({ cards: a.deadwood.slice().sort(bySuitRank), meld: -1 });
  return groups;
}

function meldIndexMap(cards) {
  const map = new Map();
  arrange(cards).melds.forEach((m, i) => m.forEach((c) => map.set(c, i)));
  return map;
}

function log(text) {
  const li = document.createElement('li');
  li.textContent = text;
  $('log').appendChild(li);
  $('log').scrollTop = 1e9;
}

function setMessage(t) { message = t; }

// ---------------------------------------------------------------------------
// main render
// ---------------------------------------------------------------------------
function render() {
  // score
  $('score').innerHTML = match
    ? `<span>あなた <b>${match.scores[HUMAN]}</b></span><span>AI <b>${match.scores[AI]}</b></span><span class="target">目標 ${match.target}</span>`
    : '';
  $('oppName').textContent = `AI（${LEVEL_NAMES[settings.level]}）`;
  document.querySelectorAll('[data-sort]').forEach((b) => b.classList.toggle('on', b.dataset.sort === settings.sort));
  if (!hand) return;

  const myTurn = hand.turn === HUMAN && hand.phase !== 'over' && !busy;
  const dealerTxt = (p) => (hand.dealer === p ? '・ディーラー' : '');

  // opponent hand
  const oppBox = $('oppHand');
  oppBox.innerHTML = '';
  oppBox.className = 'hand opp-hand';
  const showOpp = settings.showAI || hand.phase === 'over';
  const oh = hand.hands[AI];
  if (showOpp) {
    oppBox.classList.add('open');
    const mi = meldIndexMap(oh);
    for (const g of groupHandFor(oh)) g.cards.forEach((c, i) => oppBox.appendChild(cardEl(c, `${mi.has(c) ? 'm' + (mi.get(c) % 4) : ''} ${i === 0 ? 'gstart' : ''}`)));
    fitOverlap(oppBox, oh.length, '--overlap');
  } else {
    for (let i = 0; i < oh.length; i++) oppBox.appendChild(backEl());
    fitOverlap(oppBox, oh.length, '--opp-overlap');
  }
  const known = knownOppCards();
  $('oppInfo').textContent = `${oh.length}枚${dealerTxt(AI)}${known.length && !showOpp ? '・持っていると分かっている札: ' + known.map(cardName).join(' ') : ''}`;

  // piles
  $('stockCount').textContent = hand.stock.length;
  const canStock = myTurn && hand.phase === 'draw';
  const canTake = myTurn && (hand.phase === 'draw' || hand.phase === 'upcard') && !hand.mustStock && hand.discardPile.length > 0;
  $('stock').disabled = !canStock;
  $('stock').classList.toggle('active', canStock);
  const dp = $('discard');
  dp.innerHTML = '';
  if (hand.discardPile.length) dp.appendChild(cardEl(hand.top()));
  else { const e = document.createElement('div'); e.className = 'empty'; dp.appendChild(e); }
  dp.disabled = !canTake;
  dp.classList.toggle('active', canTake);

  // my hand
  const mine = hand.hands[HUMAN];
  const box = $('myHand');
  box.innerHTML = '';
  const mi = meldIndexMap(mine);
  const groups = groupHand(mine);
  const discardPhase = myTurn && hand.phase === 'discard';
  for (const g of groups) {
    g.cards.forEach((c, i) => {
      const cls = [];
      if (mi.has(c)) cls.push('m' + (mi.get(c) % 4));
      if (i === 0 && groups.length > 1) cls.push('gstart');
      if (c === freshCard) cls.push('fresh');
      if (c === hintCard) cls.push('hint');
      if (discardPhase) cls.push(c === hand.takenCard ? 'locked' : 'selectable');
      if (c === selected) cls.push('selected');
      const el = cardEl(c, cls.join(' '));
      if (discardPhase && c !== hand.takenCard) el.addEventListener('click', () => onCardClick(c));
      box.appendChild(el);
    });
  }
  fitOverlap(box, mine.length, '--overlap');

  // info line
  let info;
  if (mine.length === 11) {
    const out = new Int32Array(32);
    const mn = discardDeadwoods(mine, out);
    info = `11枚・捨てた後の最小デッドウッド ${mn}`;
    if (selected != null) info += `（${cardName(selected)}を捨てると ${deadwood(mine.filter((x) => x !== selected))}）`;
  } else {
    info = `デッドウッド ${deadwood(mine)}`;
  }
  $('myInfo').textContent = info + dealerTxt(HUMAN);

  // actions
  const act = $('actions');
  act.innerHTML = '';
  const addBtn = (label, cls, fn, disabled = false) => {
    const b = document.createElement('button');
    b.className = 'btn ' + cls; b.textContent = label; b.disabled = disabled;
    b.addEventListener('click', fn);
    act.appendChild(b);
  };
  if (myTurn) {
    if (hand.phase === 'upcard') {
      addBtn(`${cardName(hand.top())} を取る`, 'primary', () => humanAction({ type: 'take' }));
      addBtn('パス', '', () => humanAction({ type: 'pass' }));
    } else if (hand.phase === 'draw') {
      addBtn('山札から引く', 'primary', () => humanAction({ type: 'stock' }));
      if (!hand.mustStock) addBtn(`${cardName(hand.top())} を取る`, '', () => humanAction({ type: 'take' }));
    } else if (hand.phase === 'discard') {
      if (hand.canBigGin()) addBtn('ビッグジン！', 'knock', () => humanAction({ type: 'bigGin' }));
      const sel = selected;
      addBtn('捨てる', 'primary', () => humanAction({ type: 'discard', card: sel, knock: false }), sel == null);
      const dwAfter = sel == null ? 99 : deadwood(mine.filter((x) => x !== sel));
      addBtn(dwAfter === 0 ? 'ジン！' : 'ノック', 'knock', () => humanAction({ type: 'discard', card: sel, knock: true }), sel == null || dwAfter > KNOCK_LIMIT);
    }
    addBtn('ヒント', '', showHint);
  }

  const msg = $('message');
  msg.innerHTML = '';
  msg.textContent = message;
  if (hintText && myTurn) {
    const h = document.createElement('span');
    h.className = 'hintline';
    h.textContent = hintText;
    msg.appendChild(h);
  }
}

function groupHandFor(cards) {
  const a = arrange(cards);
  const groups = a.melds.map((m) => ({ cards: m }));
  if (a.deadwood.length) groups.push({ cards: a.deadwood.slice().sort(bySuitRank) });
  return groups;
}

function knownOppCards() {
  const known = new Set();
  for (const e of hand.events) {
    if (e.p !== AI) continue;
    if (e.type === 'take') known.add(e.card);
    if (e.type === 'discard') known.delete(e.card);
  }
  return [...known];
}

// ---------------------------------------------------------------------------
// flow
// ---------------------------------------------------------------------------
function startMatch() {
  token++;
  match = new GinMatch({ target: settings.target, firstDealer: Math.random() < 0.5 ? 0 : 1 });
  $('log').innerHTML = '';
  startHand();
}

function startHand() {
  hand = match.newHand();
  selected = null; freshCard = null; hintCard = null; hintText = ''; busy = false;
  log(`--- 第${match.history.length + 1}ハンド（ディーラー: ${hand.dealer === HUMAN ? 'あなた' : 'AI'}）表札 ${cardName(hand.top())}`);
  closeModal();
  promptTurn();
}

function promptTurn() {
  if (hand.phase === 'over') return finishHand();
  if (hand.turn === AI) {
    busy = true;
    setMessage('AIが考えています…');
    render();
    aiStep();
    return;
  }
  busy = false;
  if (hand.phase === 'upcard') setMessage(`表札 ${cardName(hand.top())} を取りますか？（パスすると${hand.upPasses === 0 ? '相手に権利が移ります' : '山札から引いて始めます'}）`);
  else if (hand.phase === 'draw') setMessage(hand.mustStock ? '山札から1枚引いてください。' : '山札か捨て札から1枚引いてください。');
  else if (hand.phase === 'discard') setMessage('捨てるカードを選んでください。デッドウッド10以下ならノックできます。');
  render();
}

function describe(p, a, card) {
  const who = p === HUMAN ? 'あなた' : 'AI';
  switch (a.type) {
    case 'pass': return `${who}は表札をパス`;
    case 'take': return `${who}は捨て札の ${cardName(card)} を取った`;
    case 'stock': return `${who}は山札から引いた`;
    case 'discard': return `${who}は ${cardName(a.card)} を捨てた${a.knock ? '（ノック！）' : ''}`;
    case 'bigGin': return `${who}はビッグジン！`;
    default: return '';
  }
}

function doAction(p, a) {
  let card = null;
  if (a.type === 'take') card = hand.top();
  const drawn = (() => { const r = hand.apply(a); return r; })();
  if (p === HUMAN && (a.type === 'stock' || a.type === 'take')) freshCard = a.type === 'stock' ? drawn : card;
  if (p === HUMAN && a.type === 'discard') freshCard = null;
  log(describe(p, a, card));
  return card;
}

function humanAction(a) {
  if (busy || hand.turn !== HUMAN) return;
  if (a.type === 'discard' && a.card == null) return;
  hintCard = null; hintText = ''; selected = null;
  doAction(HUMAN, a);
  promptTurn();
}

function onCardClick(c) {
  if (selected === c) humanAction({ type: 'discard', card: c, knock: false });
  else { selected = c; render(); }
}

async function aiStep() {
  const t = token;
  const view = hand.view(AI);
  const t0 = nowMs();
  const a = await askAI(view, settings.level);
  const thinkMs = nowMs() - t0;
  (window.__aiThinkMs ||= []).push(Math.round(thinkMs)); // for diagnostics / tests
  const rest = AI_WAIT / settings.speed - thinkMs;
  if (rest > 0) await sleep(rest);
  if (t !== token) return;
  const card = doAction(AI, a);
  if (a.type === 'take') setMessage(`AIは捨て札の ${cardName(card)} を取りました。`);
  else if (a.type === 'stock') setMessage('AIは山札から引きました。');
  else if (a.type === 'pass') setMessage(`AIは表札 ${cardName(view.discardPile[view.discardPile.length - 1])} をパスしました。`);
  else if (a.type === 'discard') setMessage(`AIは ${cardName(a.card)} を捨てました。${a.knock ? 'ノック！' : ''}`);
  render();
  if (hand.phase === 'over') {
    await sleep(AI_WAIT / settings.speed);
    if (t === token) finishHand();
    return;
  }
  if (hand.turn === AI) {
    aiStep();
  } else {
    busy = false;
    const last = message;
    promptTurn();
    if (hand.phase !== 'over') { setMessage(last + ' ' + message); render(); }
  }
}

async function showHint() {
  const t = token;
  const h = hand;
  hintText = 'ヒント: 考えています…';
  render();
  const v = h.view(HUMAN);
  const a = await askAI(v, 'strong');
  if (t !== token || h !== hand || hand.turn !== HUMAN || hand.phase !== v.phase || hand.hands[HUMAN].length !== v.hand.length) return;
  hintCard = null;
  if (a.type === 'take') hintText = `ヒント: 捨て札の ${cardName(hand.top())} を取る`;
  else if (a.type === 'pass') hintText = 'ヒント: パス';
  else if (a.type === 'stock') hintText = 'ヒント: 山札から引く';
  else if (a.type === 'bigGin') hintText = 'ヒント: ビッグジン！';
  else if (a.type === 'discard') {
    hintCard = a.card;
    hintText = `ヒント: ${cardName(a.card)} を${a.knock ? '捨ててノック' : '捨てる'}`;
  }
  render();
}

// ---------------------------------------------------------------------------
// results
// ---------------------------------------------------------------------------
function handBlock(title, arr, layoffs, knockerMelds) {
  const wrap = document.createElement('div');
  const h = document.createElement('h3');
  h.textContent = title;
  wrap.appendChild(h);
  const row = document.createElement('div');
  row.className = 'result-hand';
  const grp = (cards, cap, cls = '') => {
    const g = document.createElement('div'); g.className = 'grp';
    const cs = document.createElement('div'); cs.className = 'cards';
    cards.forEach((c) => cs.appendChild(cardEl(c, cls)));
    const cp = document.createElement('div'); cp.className = 'cap'; cp.textContent = cap;
    g.appendChild(cs); g.appendChild(cp); row.appendChild(g);
  };
  arr.melds.forEach((m, i) => grp(m, 'メルド', 'm' + (i % 4)));
  (layoffs || []).forEach((l) => grp(l.cards, `付け札→${knockerMelds[l.target].map(cardName).join('')}`, 'layoff'));
  if (arr.deadwood.length) grp(arr.deadwood, `デッドウッド ${arr.deadwood.reduce((s, c) => s + Math.min(rankOf(c) + 1, 10), 0)}`);
  wrap.appendChild(row);
  return wrap;
}

function finishHand() {
  const r = hand.result;
  busy = true;
  render();
  match.recordResult(r);
  const body = $('modalBody');
  body.innerHTML = '';
  const h2 = document.createElement('h2');
  const who = (p) => (p === HUMAN ? 'あなた' : 'AI');
  if (r.type === 'draw') {
    h2.textContent = '流局（山札切れ）';
    body.appendChild(h2);
    const p = document.createElement('p'); p.textContent = '山札が残り2枚になったため、このハンドは無得点です。';
    body.appendChild(p);
  } else {
    const titles = {
      knock: `${who(r.knocker)}のノック`,
      gin: `${who(r.knocker)}のジン！`,
      biggin: `${who(r.knocker)}のビッグジン！`,
      undercut: `アンダーカット！（${who(r.winner)}の勝ち）`,
    };
    h2.textContent = (r.winner === HUMAN ? '🎉 ' : '') + titles[r.type];
    body.appendChild(h2);
    body.appendChild(handBlock(`${who(r.knocker)}（${r.type === 'knock' || r.type === 'undercut' ? 'ノック' : 'ジン'}・デッドウッド ${r.knockerDW}）`, r.knockerArr, null, null));
    body.appendChild(handBlock(`${who(1 - r.knocker)}（デッドウッド ${r.defenderDW}）`, r.defenderArr, r.defenderArr.layoffs, r.knockerArr.melds));
    const p = document.createElement('p');
    p.className = 'points';
    let detail;
    if (r.type === 'knock') detail = `${r.defenderDW} − ${r.knockerDW}`;
    else if (r.type === 'undercut') detail = `アンダーカット ${r.bonus} ＋ (${r.knockerDW} − ${r.defenderDW})`;
    else detail = `${r.type === 'biggin' ? 'ビッグジン' : 'ジン'} ${r.bonus} ＋ ${r.defenderDW}`;
    p.innerHTML = `${who(r.winner)}に <b>${r.points}</b> 点（${detail}）`;
    body.appendChild(p);
  }
  const sc = document.createElement('p');
  sc.textContent = `合計: あなた ${match.scores[HUMAN]} − AI ${match.scores[AI]}（目標 ${match.target}）`;
  body.appendChild(sc);
  log(r.type === 'draw' ? '流局' : `${who(r.winner)} +${r.points}（${r.type}）`);

  const actions = $('modalActions');
  actions.innerHTML = '';
  if (match.final) {
    addModalBtn('最終結果へ', 'primary', showFinal);
  } else {
    addModalBtn('次のハンドへ', 'primary', startHand);
  }
  openModal();
}

function showFinal() {
  const f = match.final;
  const body = $('modalBody');
  body.innerHTML = '';
  const h2 = document.createElement('h2');
  h2.textContent = f.winner === HUMAN ? '🏆 あなたの勝ち！' : 'AIの勝ち…';
  body.appendChild(h2);
  const t = document.createElement('table');
  t.className = 'final-table';
  const row = (label, a, b) => `<tr><td>${label}</td><td>${a}</td><td>${b}</td></tr>`;
  t.innerHTML = `<tr><th></th><th>あなた</th><th>AI</th></tr>`
    + row('ハンド得点', match.scores[HUMAN], match.scores[AI])
    + row(`ボックス（勝ったハンド×${BOX_BONUS}）`, f.box[HUMAN], f.box[AI])
    + row('ゲームボーナス', f.winner === HUMAN ? GAME_BONUS : 0, f.winner === AI ? GAME_BONUS : 0)
    + (f.shutout ? row('シャットアウト', f.winner === HUMAN ? '×2' : '', f.winner === AI ? '×2' : '') : '')
    + row('<b>合計</b>', `<b>${f.totals[HUMAN]}</b>`, `<b>${f.totals[AI]}</b>`);
  body.appendChild(t);
  // stats
  const key = settings.level;
  const s = stats[key] || { w: 0, l: 0 };
  if (f.winner === HUMAN) s.w++; else s.l++;
  stats[key] = s;
  store.set('stats2', stats);
  const st = document.createElement('p');
  st.className = 'stats';
  st.textContent = `対「${LEVEL_NAMES[key]}」通算: ${s.w}勝 ${s.l}敗`;
  body.appendChild(st);
  const actions = $('modalActions');
  actions.innerHTML = '';
  addModalBtn('もう一度対戦', 'primary', startMatch);
}

function addModalBtn(label, cls, fn) {
  const b = document.createElement('button');
  b.className = 'btn ' + cls; b.textContent = label;
  b.addEventListener('click', fn);
  $('modalActions').appendChild(b);
}
function openModal() { $('modal').hidden = false; }
function closeModal() { $('modal').hidden = true; }

// ---------------------------------------------------------------------------
// dialogs
// ---------------------------------------------------------------------------
function showRules() {
  const body = $('modalBody');
  body.innerHTML = '';
  body.appendChild($('rulesTpl').content.cloneNode(true));
  $('modalActions').innerHTML = '';
  addModalBtn('閉じる', 'primary', closeModal);
  openModal();
}

function showSettings() {
  const body = $('modalBody');
  body.innerHTML = `
    <h2>設定</h2>
    <div class="settings">
      <label>AIの強さ
        <select id="setLevel">
          <option value="strong">最強</option>
          <option value="normal">ふつう</option>
          <option value="easy">よわい</option>
        </select>
      </label>
      <label>目標点（次のゲームから）
        <select id="setTarget">
          <option value="100">100</option><option value="150">150</option><option value="250">250</option><option value="50">50（短め）</option>
        </select>
      </label>
      <label>AIの速さ
        <select id="setSpeed"><option value="0.6">ゆっくり</option><option value="1">ふつう</option><option value="2.5">はやい</option></select>
      </label>
      <label>AIの手札を表示（練習用）
        <input type="checkbox" id="setShow">
      </label>
    </div>
    <p class="stats">通算成績: ${Object.entries(LEVEL_NAMES).map(([k, n]) => `${n} ${(stats[k] || { w: 0 }).w}勝${(stats[k] || { l: 0 }).l}敗`).join(' / ')}</p>
    <p class="stats">「最強」AIは、出たカードを全て記憶して相手の手札を推測し、ありうる相手の手札と山札の並びを数百通り仮定して相手の次の手番までシミュレーションします。その局面を、AI同士の数万局の対戦から学習した評価式で採点して、最も得点が期待できる手を選びます（1手あたり約0.2秒以内）。</p>`;
  $('setLevel').value = settings.level;
  $('setTarget').value = String(settings.target);
  $('setSpeed').value = String(settings.speed);
  $('setShow').checked = settings.showAI;
  const save = () => {
    settings.level = $('setLevel').value;
    settings.target = +$('setTarget').value;
    settings.speed = +$('setSpeed').value;
    settings.showAI = $('setShow').checked;
    store.set('settings', settings);
    render();
  };
  body.querySelectorAll('select, input').forEach((el) => el.addEventListener('change', save));
  $('modalActions').innerHTML = '';
  addModalBtn('閉じる', 'primary', closeModal);
  openModal();
}

// modal close must resume a finished hand's dialog if needed
function closeModalSafe() {
  closeModal();
  if (hand && hand.phase === 'over' && match.history[match.history.length - 1] === hand.result) {
    // result dialog was replaced by rules/settings: reopen it
    reopenResult();
  }
}
function reopenResult() {
  const actions = $('modalActions');
  const body = $('modalBody');
  body.innerHTML = '<h2>ハンド終了</h2>';
  const sc = document.createElement('p');
  sc.textContent = `合計: あなた ${match.scores[HUMAN]} − AI ${match.scores[AI]}（目標 ${match.target}）`;
  body.appendChild(sc);
  actions.innerHTML = '';
  if (match.final) addModalBtn('最終結果へ', 'primary', showFinal);
  else addModalBtn('次のハンドへ', 'primary', startHand);
  openModal();
}

// ---------------------------------------------------------------------------
// wiring
// ---------------------------------------------------------------------------
$('stock').addEventListener('click', () => humanAction({ type: 'stock' }));
$('discard').addEventListener('click', () => humanAction({ type: 'take' }));
$('btnNew').addEventListener('click', () => {
  if (match && !match.final && match.history.length + (hand && hand.phase !== 'over' ? 1 : 0) > 0 && !confirm('現在のゲームを終了して新しいゲームを始めますか？')) return;
  startMatch();
});
$('btnRules').addEventListener('click', () => { showRules(); setCloser(); });
$('btnSettings').addEventListener('click', () => { showSettings(); setCloser(); });
function setCloser() {
  const btn = $('modalActions').querySelector('button');
  if (btn) { btn.replaceWith(btn.cloneNode(true)); $('modalActions').querySelector('button').addEventListener('click', closeModalSafe); }
}
document.querySelectorAll('[data-sort]').forEach((b) => b.addEventListener('click', () => {
  settings.sort = b.dataset.sort; store.set('settings', settings); render();
}));
window.addEventListener('resize', () => render());

startMatch();
if (!store.get('seenRules', false)) { store.set('seenRules', true); showRules(); setCloser(); }
