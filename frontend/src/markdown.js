// Tiny, safe markdown renderer for agent replies.
// It escapes ALL HTML first, then introduces only a small set of tags we
// control (strong/em/code/ul/ol/li/p) — so model output can never inject HTML.

function escapeHtml(s) {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
}

function inline(t) {
  return t
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/(^|[^*])\*(?!\s)([^*]+?)\*/g, '$1<em>$2</em>')
}

export function renderMarkdown(src) {
  const lines = escapeHtml(src || '').split('\n')
  const out = []
  let list = null // 'ul' | 'ol' | null

  const closeList = () => {
    if (list) { out.push(`</${list}>`); list = null }
  }

  for (const line of lines) {
    const ordered = line.match(/^\s*\d+\.\s+(.*)$/)
    const bullet = line.match(/^\s*[-*]\s+(.*)$/)
    if (ordered || bullet) {
      const want = ordered ? 'ol' : 'ul'
      if (list !== want) { closeList(); out.push(`<${want}>`); list = want }
      out.push('<li>' + inline((ordered || bullet)[1]) + '</li>')
    } else if (line.trim() === '') {
      closeList()
    } else {
      closeList()
      out.push('<p>' + inline(line) + '</p>')
    }
  }
  closeList()
  return out.join('\n')
}
