import { useState, useEffect, useRef } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { fetchRecipe, toggleFavorite, uploadImage, askRecipeAI, deleteRecipe } from '../api/recipeApi'
import BottomNav from '../components/BottomNav'
import '../global.css'

const SERVING_OPTIONS = [0.5, 1, 2, 3, 4, 6]

/* 数値を見やすい文字列に変換 */
function fmtAmt(val) {
  if (val <= 0)            return '0'
  if (val < 0.1)           return val.toFixed(2).replace(/\.?0+$/, '')
  if (Number.isInteger(val)) return String(val)
  const f = val.toFixed(1)
  return f.endsWith('.0') ? f.slice(0, -2) : f
}

/* ── AIチャットパネル ── */
function AIPanel({ recipe, onClose }) {
  const [messages, setMessages] = useState([
    { role: 'bot', text: `「${recipe.title}」について何でも聞いてください！` }
  ])
  const [input,   setInput]   = useState('')
  const [loading, setLoading] = useState(false)
  const chatRef = useRef()

  const hints = ['時短テクニックは？', 'みりんの代用は？', '失敗しないコツは？']

  const send = async (q) => {
    const text = q || input.trim()
    if (!text || loading) return
    setMessages(m => [...m, { role: 'user', text }])
    setInput('')
    setLoading(true)
    try {
      const res = await askRecipeAI(recipe.id, text)
      setMessages(m => [...m, { role: 'bot', text: res.answer }])
    } catch {
      setMessages(m => [...m, { role: 'bot', text: 'エラーが発生しました。再度お試しください。' }])
    } finally {
      setLoading(false)
      setTimeout(() => chatRef.current?.scrollTo(0, 9999), 50)
    }
  }

  return (
    <div style={{
      position: 'fixed', bottom: 64, left: '50%', transform: 'translateX(-50%)',
      width: '100%', maxWidth: 430,
      background: 'var(--surface)', borderTop: '1px solid var(--border)',
      zIndex: 200, padding: '12px 16px',
      boxShadow: '0 -4px 20px rgba(0,0,0,.10)',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
        <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--blue-500)', display: 'flex', alignItems: 'center', gap: 5 }}>
          <i className="ti ti-sparkles" />AI アシスタント
        </span>
        <button onClick={onClose} style={{ background: 'none', border: 'none', cursor: 'pointer', fontSize: 18, color: 'var(--text-3)' }}>
          <i className="ti ti-x" />
        </button>
      </div>
      {/* ヒントチップ */}
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 8 }}>
        {hints.map(h => (
          <button key={h} onClick={() => send(h)}
            style={{
              fontSize: 11, padding: '4px 10px', borderRadius: 999,
              border: '1px solid var(--border)', background: 'var(--bg)',
              color: 'var(--text-2)', cursor: 'pointer',
            }}>{h}</button>
        ))}
      </div>
      {/* チャット */}
      <div ref={chatRef} style={{
        maxHeight: 130, overflowY: 'auto',
        display: 'flex', flexDirection: 'column', gap: 6, marginBottom: 8,
        scrollbarWidth: 'thin',
      }}>
        {messages.map((m, i) => (
          <div key={i} style={{
            padding: '7px 11px', borderRadius: 10, fontSize: 13,
            lineHeight: 1.6, maxWidth: '88%', whiteSpace: 'pre-wrap',
            background: m.role === 'bot' ? 'var(--bg)' : 'var(--blue-500)',
            color:      m.role === 'bot' ? 'var(--text-1)' : '#fff',
            alignSelf:  m.role === 'bot' ? 'flex-start' : 'flex-end',
          }}>{m.text}</div>
        ))}
        {loading && (
          <div style={{ padding: '7px 11px', borderRadius: 10, background: 'var(--bg)', fontSize: 13, color: 'var(--text-3)' }}>
            回答中…
          </div>
        )}
      </div>
      {/* 入力欄 */}
      <div style={{ display: 'flex', gap: 8 }}>
        <input
          value={input} onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && send()}
          placeholder="質問を入力…"
          style={{
            flex: 1, padding: '8px 11px', borderRadius: 'var(--radius-sm)',
            border: '1px solid var(--border)', fontSize: 13,
            background: 'var(--bg)', color: 'var(--text-1)', outline: 'none',
          }}
        />
        <button onClick={() => send()}
          style={{
            padding: '8px 14px', background: 'var(--blue-500)', color: '#fff',
            border: 'none', borderRadius: 'var(--radius-sm)', fontSize: 13, cursor: 'pointer',
          }}>送信</button>
      </div>
    </div>
  )
}

/* ── メイン ── */
export default function RecipeDetailPage() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [recipe,      setRecipe]      = useState(null)
  const [loading,     setLoading]     = useState(true)
  const [servings,    setServings]    = useState(null)  // null = 未取得
  const [activeTab,   setActiveTab]   = useState('ingredients')
  const [checkedSteps,setCheckedSteps]= useState(new Set())
  const [showAI,      setShowAI]      = useState(false)
  const [showDelConf, setShowDelConf] = useState(false)

  useEffect(() => {
    fetchRecipe(id)
      .then(r => { setRecipe(r); setServings(r.base_servings) })
      .catch(() => navigate('/recipes'))
      .finally(() => setLoading(false))
  }, [id])

  const handleFav = async () => {
    const updated = await toggleFavorite(recipe.id)
    setRecipe(updated)
  }

  const handleImageUpload = async (e) => {
    const file = e.target.files[0]
    if (!file) return
    const updated = await uploadImage(recipe.id, file)
    setRecipe(updated)
  }

  const handleDelete = async () => {
    await deleteRecipe(recipe.id)
    navigate('/recipes')
  }

  const toggleStep = (order) => {
    setCheckedSteps(prev => {
      const n = new Set(prev)
      n.has(order) ? n.delete(order) : n.add(order)
      return n
    })
  }

  if (loading) return <div className="spinner"><i className="ti ti-loader-2" style={{ fontSize: 24 }} />読み込み中…</div>
  if (!recipe)  return null

  const ratio = servings / recipe.base_servings
  const servingChanged = servings !== recipe.base_servings

  return (
    <div className="page-wrapper">

      {/* ── ヒーロー画像 ── */}
      <div style={{ position: 'relative', height: 220 }}>
        {recipe.image_url ? (
          <img src={recipe.image_url} alt={recipe.title}
            style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
        ) : (
          <div style={{
            width: '100%', height: '100%', background: 'var(--blue-50)',
            display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 72,
          }}>
            <i className="ti ti-tools-kitchen-2" style={{ color: 'var(--blue-200)' }} />
          </div>
        )}
        {/* 戻るボタン */}
        <button onClick={() => navigate('/recipes')}
          style={{
            position: 'absolute', top: 12, left: 12,
            width: 36, height: 36, borderRadius: '50%',
            background: 'rgba(255,255,255,.9)', border: 'none',
            cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 18,
          }}>
          <i className="ti ti-arrow-left" />
        </button>
        {/* お気に入り */}
        <button onClick={handleFav}
          style={{
            position: 'absolute', top: 12, right: 12,
            width: 36, height: 36, borderRadius: '50%',
            background: 'rgba(255,255,255,.9)', border: 'none',
            cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 18, color: recipe.is_favorite ? 'var(--red)' : 'var(--text-3)',
          }}>
          <i className={recipe.is_favorite ? 'ti ti-heart-filled' : 'ti ti-heart'} />
        </button>
        {/* 写真変更 */}
        <label style={{
          position: 'absolute', bottom: 10, right: 10,
          background: 'rgba(0,0,0,.45)', color: '#fff',
          padding: '5px 10px', borderRadius: 'var(--radius-sm)',
          fontSize: 12, cursor: 'pointer',
          display: 'flex', alignItems: 'center', gap: 5,
        }}>
          <i className="ti ti-camera" />写真を変更
          <input type="file" accept="image/*" style={{ display: 'none' }} onChange={handleImageUpload} />
        </label>
      </div>

      {/* ── ボディ ── */}
      <div style={{ padding: '16px 20px' }}>

        {/* カテゴリ・タイトル */}
        <span style={{
          fontSize: 11, padding: '3px 10px', borderRadius: 999,
          background: 'var(--blue-50)', color: 'var(--blue-500)', fontWeight: 600,
        }}>{recipe.category}</span>
        <h1 style={{ fontSize: 22, fontWeight: 600, margin: '8px 0 14px', lineHeight: 1.25 }}>
          {recipe.title}
        </h1>
        {recipe.description && (
          <p style={{ fontSize: 14, color: 'var(--text-2)', lineHeight: 1.7, marginBottom: 14 }}>
            {recipe.description}
          </p>
        )}

        {/* ── 人数セレクター ── */}
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          background: 'var(--blue-50)', border: '1px solid var(--blue-100)',
          borderRadius: 'var(--radius-sm)', padding: '10px 14px', marginBottom: 12,
        }}>
          <div style={{ fontSize: 13, color: 'var(--text-2)', display: 'flex', alignItems: 'center', gap: 6 }}>
            <i className="ti ti-users" style={{ color: 'var(--blue-500)' }} />
            <span>人数を変更</span>
          </div>
          <select
            value={servings}
            onChange={e => setServings(parseFloat(e.target.value))}
            style={{
              border: '1px solid var(--blue-200)', borderRadius: 'var(--radius-sm)',
              padding: '5px 10px', fontSize: 14, fontWeight: 600,
              color: 'var(--blue-500)', background: 'var(--surface)',
              cursor: 'pointer', outline: 'none',
            }}
          >
            {SERVING_OPTIONS.map(s => (
              <option key={s} value={s}>{s}人前</option>
            ))}
          </select>
        </div>

        {/* 換算中バッジ */}
        {servingChanged && (
          <div style={{
            fontSize: 12, color: 'var(--blue-500)',
            background: 'var(--blue-50)', border: '1px solid var(--blue-100)',
            borderRadius: 'var(--radius-sm)', padding: '5px 12px',
            marginBottom: 12, display: 'flex', alignItems: 'center', gap: 6,
          }}>
            <i className="ti ti-refresh" style={{ fontSize: 13 }} />
            {recipe.base_servings}人前 → {servings}人前に換算済み
          </div>
        )}

        {/* 時間メタ */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 16 }}>
          {[
            { label: '下準備', value: `${recipe.prep_time}分` },
            { label: '調理', value: `${recipe.cook_time}分` },
          ].map(item => (
            <div key={item.label} style={{
              background: 'var(--bg)', borderRadius: 'var(--radius-sm)',
              padding: 10, textAlign: 'center', border: '1px solid var(--border)',
            }}>
              <div style={{ fontSize: 18, fontWeight: 600 }}>{item.value}</div>
              <div style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 2 }}>{item.label}</div>
            </div>
          ))}
        </div>

        {/* ── タブ ── */}
        <div style={{ display: 'flex', borderBottom: '1px solid var(--border)', marginBottom: 14 }}>
          {[
            { key: 'ingredients', label: `材料（${recipe.ingredients.length}種）` },
            { key: 'steps',       label: `作り方（${recipe.steps.length}工程）` },
          ].map(tab => (
            <button key={tab.key} onClick={() => setActiveTab(tab.key)}
              style={{
                flex: 1, padding: '9px 0', fontSize: 13, fontWeight: 600,
                background: 'none', border: 'none', cursor: 'pointer',
                color: activeTab === tab.key ? 'var(--blue-500)' : 'var(--text-3)',
                borderBottom: activeTab === tab.key ? '2px solid var(--blue-500)' : '2px solid transparent',
                marginBottom: -1, transition: 'all var(--t)',
              }}>{tab.label}</button>
          ))}
        </div>

        {/* ── 材料タブ ── */}
        {activeTab === 'ingredients' && (
          <div>
            {recipe.ingredients.map((ing, i) => {
              const scaled = ing.amount * ratio
              return (
                <div key={i} style={{
                  display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                  padding: '11px 0', borderBottom: i < recipe.ingredients.length - 1 ? '1px solid var(--border)' : 'none',
                  fontSize: 14,
                }}>
                  <span>{ing.name}</span>
                  <span style={{ fontWeight: 600, color: 'var(--blue-500)', minWidth: 80, textAlign: 'right' }}>
                    {fmtAmt(scaled)} {ing.unit}
                  </span>
                </div>
              )
            })}
          </div>
        )}

        {/* ── 手順タブ ── */}
        {activeTab === 'steps' && (
          <div>
            <p style={{ fontSize: 12, color: 'var(--text-3)', marginBottom: 10 }}>
              ステップをタップして完了をマーク
            </p>
            {recipe.steps.map(step => {
              const done = checkedSteps.has(step.order)
              return (
                <div key={step.order}
                  onClick={() => toggleStep(step.order)}
                  style={{
                    display: 'flex', gap: 12, padding: '12px 0',
                    borderBottom: `1px solid var(--border)`,
                    cursor: 'pointer', opacity: done ? .5 : 1,
                    transition: 'opacity var(--t)',
                  }}
                >
                  <div style={{
                    width: 28, height: 28, borderRadius: '50%', flexShrink: 0,
                    background: done ? 'var(--text-3)' : 'var(--blue-500)',
                    color: '#fff', fontSize: 12, fontWeight: 600,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    marginTop: 1, transition: 'background var(--t)',
                  }}>
                    {done ? <i className="ti ti-check" style={{ fontSize: 13 }} /> : step.order}
                  </div>
                  <div style={{ flex: 1 }}>
                    <p style={{
                      fontSize: 14, lineHeight: 1.7,
                      textDecoration: done ? 'line-through' : 'none',
                    }}>{step.description}</p>
                    {step.tip && (
                      <div style={{
                        background: '#fffbeb', borderLeft: '3px solid var(--gold)',
                        borderRadius: '0 6px 6px 0', padding: '6px 10px',
                        fontSize: 12, color: '#78350f', marginTop: 6,
                        display: 'flex', gap: 6,
                      }}>
                        <i className="ti ti-bulb" style={{ fontSize: 14, flexShrink: 0 }} />
                        {step.tip}
                      </div>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        )}

        {/* ── アクションボタン行 ── */}
        <div style={{ display: 'flex', gap: 8, marginTop: 20, flexWrap: 'wrap' }}>
          <button className="btn btn-primary" onClick={() => navigate(`/recipes/${id}/edit`)}>
            <i className="ti ti-edit" />編集
          </button>
          <button className="btn btn-ghost" onClick={() => setShowAI(v => !v)}>
            <i className="ti ti-sparkles" />AI相談
          </button>
          <button className="btn btn-danger" onClick={() => setShowDelConf(true)}>
            <i className="ti ti-trash" />削除
          </button>
        </div>

        {/* 削除確認 */}
        {showDelConf && (
          <div style={{
            position: 'fixed', inset: 0, background: 'rgba(0,0,0,.4)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            zIndex: 300, padding: 20,
          }}>
            <div style={{
              background: 'var(--surface)', borderRadius: 'var(--radius-lg)',
              padding: 24, maxWidth: 320, width: '100%',
            }}>
              <p style={{ fontWeight: 600, marginBottom: 8 }}>「{recipe.title}」を削除しますか？</p>
              <p style={{ fontSize: 13, color: 'var(--text-3)', marginBottom: 20 }}>
                この操作は取り消せません。
              </p>
              <div style={{ display: 'flex', gap: 8 }}>
                <button className="btn btn-ghost" style={{ flex: 1 }}
                  onClick={() => setShowDelConf(false)}>キャンセル</button>
                <button className="btn btn-danger" style={{ flex: 1 }}
                  onClick={handleDelete}>削除する</button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* ── AIパネル ── */}
      {showAI && <AIPanel recipe={recipe} onClose={() => setShowAI(false)} />}

      <BottomNav />
    </div>
  )
}
