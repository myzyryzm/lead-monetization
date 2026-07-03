import React, { useEffect, useRef, useState } from 'react'
import { sendChat } from '../api.js'
import { renderMarkdown } from '../markdown.js'

const EXAMPLES = [
  "We signed an auto-insurance offer at $22/lead — who do we send to and what's the projected revenue from 500 sends?",
  'Best offers for a finance lead from google search who last opened 5 days ago?',
  'Draft an email and SMS for a $30 health paid-trial offer.',
  'For a $6 sweepstakes email-submit offer, who clears a $0.50/lead EV floor?',
]

export default function Chat() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const endRef = useRef(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, busy])

  async function send(text) {
    const content = (text ?? input).trim()
    if (!content || busy) return
    setError(null)
    const next = [...messages, { role: 'user', content }]
    setMessages(next)
    setInput('')
    setBusy(true)
    try {
      const data = await sendChat(next)
      setMessages(data.messages)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <section className="chat">
      <div className="messages">
        {messages.length === 0 && (
          <div className="empty">
            <p>
              Describe an offer campaign and I'll tell you who to send it to, the
              projected revenue, and draft the copy. Try one of these:
            </p>
            <div className="examples">
              {EXAMPLES.map((ex, i) => (
                <button key={i} className="example" onClick={() => send(ex)}>
                  {ex}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            {m.role === 'assistant' ? (
              <div
                className="bubble"
                dangerouslySetInnerHTML={{ __html: renderMarkdown(m.content) }}
              />
            ) : (
              <div className="bubble">{m.content}</div>
            )}
          </div>
        ))}

        {busy && (
          <div className="msg assistant">
            <div className="bubble thinking">thinking…</div>
          </div>
        )}
        {error && (
          <div className="msg error">
            <div className="bubble">Error: {error}</div>
          </div>
        )}
        <div ref={endRef} />
      </div>

      <form className="composer" onSubmit={(e) => { e.preventDefault(); send() }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="e.g. We signed a $22 auto-insurance offer — who do we send it to?"
          disabled={busy}
        />
        <button type="submit" disabled={busy || !input.trim()}>Send</button>
      </form>
    </section>
  )
}
