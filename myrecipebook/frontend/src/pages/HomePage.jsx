import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { fetchRecipes, suggestMenu } from '../api/recipeApi'
import BottomNav from '../components/BottomNav'
import '../global.css'

// トレンド（オンライン化後に実APIに差し替え）
const MOCK_TRENDS = [
  { title: '親子丼',      heat: '1.2k', img: 'https://images.unsplash.com/photo-1512058564366-18510be2db19?w=180&q=60' },
  { title: '醤油ラーメン', heat: '980',  img: 'https://images.unsplash.com/photo-1569050467447-ce54b3bbc37d?w=180&q=60' },
  { title: 'チャーハン',   heat: '741',  img: 'https://images.unsplash.com/photo-1603133872878-684f208fb84b?w=180&q=60' },
]

function timeGreeting() {
  const h = new Date().getHours()
  if (h < 11) return 'おはようございます'
  if (h < 17) return 'こんにちは'
  return 'こんばんは'
}

export default function HomePage() {
  const navigate = useNavigate()
  const [recipes,    setRecipes]    = useState([])
  const [aiSuggest,  setAiSuggest]  = useState(null)
  const [aiLoading,  setAiLoading]  = useState(false)

  useEffect(() => {
    fetchRecipes({ sort: 'updated_at', order: 'desc' })
      .then(r => {
        setRecipes(r)
        if (r.length > 0) pickAiSuggest(r)
      })
      .catch(() => {})
  }, [])

  // ライブラリから簡易サジェスト（時間帯ベース、オンライン化後はRAGに切り替え）
  const pickAiSuggest = (list) => {
    const h = new Date().getHours()
    const scored = list.map(r => ({
      ...r,
      score: (h >= 17 && r.cook_time > 20 ? 2 : 0) +
             (h < 12 && r.cook_time < 20   ? 2 : 0) +
             (r.is_favorite               ? 1 : 0),
    }))
    scored.sort((a, b) => b.score - a.score)
    setAiSuggest(scored[0])
  }

  const handleAskAI = async () => {
    if (recipes.length === 0) return
    setAiLoading(true)
    try {
      const h = new Date().getHours()
      const q = h >= 17 ? '今夜の夕食に何を作ればいいですか？' : '今日のランチに何を作ればいいですか？'
      await suggestMenu(q)
    } catch {}
    finally { setAiLoading(false) }
  }

  const recent = recipes.slice(0, 3)

  return (
    <div className="page-wrapper">
      {/* ── ハニーゴールドヘッダー ── */}
      <div className="topbar">
        <div className="topbar-row">
          <div>
            <div className="topbar-title">🍳 MyRecipe</div>
            <div className="topbar-sub">{timeGreeting()}</div>
          </div>
          <div style={{
            width: 34, height: 34, borderRadius: '50%',
            background: 'rgba(255,255,255,.3)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 13, fontWeight: 600, color: '#fff',
          }}>田</div>
        </div>
      </div>

      {/* ── AIおすすめ ── */}
      {aiSuggest && (
        <>
          <div className="section-label">
            <i className="ti ti-sparkles" style={{ color: 'var(--gold-dark)' }} aria-hidden="true" />
            今夜のAIおすすめ
          </div>
          <div style={{ padding: '0 16px' }}>
            <div
              onClick={() => navigate(`/recipes/${aiSuggest.id}`)}
              style={{
                background: 'var(--gold-light)', border: '1px solid #E8D080',
                borderRadius: 'var(--radius-md)', display: 'flex',
                gap: 12, padding: 14, cursor: 'pointer',
                transition: 'box-shadow var(--t)',
              }}
              onMouseEnter={e => e.currentTarget.style.boxShadow = 'var(--shadow-md)'}
              onMouseLeave={e => e.currentTarget.style.boxShadow = ''}
            >
              <div style={{
                width: 70, height: 70, borderRadius: 'var(--radius-sm)',
                overflow: 'hidden', flexShrink: 0,
                background: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}>
                {aiSuggest.image_url
                  ? <img src={aiSuggest.image_url} alt={aiSuggest.title}
                      style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                  : <i className="ti ti-tools-kitchen-2" style={{ fontSize: 30, color: 'var(--gold-dark)', opacity: .6 }} />
                }
              </div>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{
                  fontSize: 11, color: 'var(--gold-dark)', fontWeight: 600,
                  display: 'flex', alignItems: 'center', gap: 4, marginBottom: 4,
                }}>
                  <i className="ti ti-brain" style={{ fontSize: 12 }} aria-hidden="true" />
                  あなたの好みから推測
                </div>
                <div style={{ fontSize: 15, fontWeight: 600, color: 'var(--text-1)' }}>
                  {aiSuggest.title}
                </div>
                <div style={{ fontSize: 12, color: 'var(--text-3)', marginTop: 3 }}>
                  <i className="ti ti-clock" style={{ fontSize: 11 }} /> {aiSuggest.cook_time}分　
                  <i className="ti ti-users" style={{ fontSize: 11 }} /> {aiSuggest.base_servings}人前
                </div>
                <div style={{
                  marginTop: 6, fontSize: 12, color: 'var(--blue)',
                  display: 'flex', alignItems: 'center', gap: 3,
                }}>
                  <i className="ti ti-arrow-right" style={{ fontSize: 12 }} aria-hidden="true" />詳細を見る
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      {/* ライブラリ登録ゼロの場合 */}
      {recipes.length === 0 && (
        <div style={{ padding: '40px 20px', textAlign: 'center' }}>
          <div style={{
            width: 64, height: 64, borderRadius: '50%',
            background: 'var(--gold-light)', margin: '0 auto 12px',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <i className="ti ti-tools-kitchen-2" style={{ fontSize: 30, color: 'var(--gold-dark)' }} />
          </div>
          <p style={{ fontSize: 15, fontWeight: 600, marginBottom: 6 }}>レシピを追加してみましょう</p>
          <p style={{ fontSize: 13, color: 'var(--text-3)', lineHeight: 1.7, marginBottom: 16 }}>
            レシピを登録するとAIがあなたの好みに合わせて<br />今夜の献立を提案します
          </p>
          <button className="btn btn-gold" onClick={() => navigate('/recipes/new')}>
            <i className="ti ti-plus" />最初のレシピを追加
          </button>
        </div>
      )}

      {/* ── トレンド（オンライン時） ── */}
      <div className="section-label" style={{ marginTop: 8 }}>
        <i className="ti ti-trending-up" style={{ color: '#E24B4A' }} aria-hidden="true" />
        今週のトレンド
        <span style={{
          fontSize: 10, background: '#FCEBEB', color: '#A32D2D',
          padding: '2px 6px', borderRadius: 4, marginLeft: 4,
        }}>オンライン化後に有効</span>
      </div>
      <div style={{ display: 'flex', gap: 10, padding: '0 16px', overflowX: 'auto', scrollbarWidth: 'none' }}>
        {MOCK_TRENDS.map((t, i) => (
          <div key={i} style={{
            flexShrink: 0, width: 110,
            background: 'var(--surface)', border: '1px solid var(--border)',
            borderRadius: 'var(--radius-md)', overflow: 'hidden',
            opacity: .55,
          }}>
            <img src={t.img} alt={t.title}
              style={{ width: '100%', height: 72, objectFit: 'cover', display: 'block' }} />
            <div style={{ padding: '6px 8px' }}>
              <div style={{ fontSize: 11, fontWeight: 600 }}>{t.title}</div>
              <div style={{ fontSize: 10, color: '#E24B4A', marginTop: 2, display: 'flex', alignItems: 'center', gap: 2 }}>
                <i className="ti ti-flame" style={{ fontSize: 11 }} aria-hidden="true" />{t.heat}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* ── 最近見たレシピ ── */}
      {recent.length > 0 && (
        <>
          <div className="section-label" style={{ marginTop: 10 }}>
            <i className="ti ti-clock" style={{ color: 'var(--text-3)' }} aria-hidden="true" />
            最近追加したレシピ
          </div>
          <div style={{ display: 'flex', gap: 10, padding: '0 16px', overflowX: 'auto', scrollbarWidth: 'none' }}>
            {recent.map(r => (
              <div
                key={r.id}
                onClick={() => navigate(`/recipes/${r.id}`)}
                style={{
                  flexShrink: 0, width: 130,
                  background: 'var(--surface)', border: '1px solid var(--border)',
                  borderRadius: 'var(--radius-md)', overflow: 'hidden', cursor: 'pointer',
                  transition: 'box-shadow var(--t)',
                }}
                onMouseEnter={e => e.currentTarget.style.boxShadow = 'var(--shadow-md)'}
                onMouseLeave={e => e.currentTarget.style.boxShadow = ''}
              >
                <div style={{ height: 80, background: 'var(--gold-light)', overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  {r.image_url
                    ? <img src={r.image_url} alt={r.title} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                    : <i className="ti ti-tools-kitchen-2" style={{ fontSize: 32, color: 'var(--gold-dark)', opacity: .6 }} />
                  }
                </div>
                <div style={{ padding: '7px 9px' }}>
                  <div style={{ fontSize: 12, fontWeight: 600, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {r.title}
                  </div>
                  <span className={`cat-badge cat-${r.category}`} style={{ marginTop: 4, fontSize: 10 }}>
                    {r.category}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </>
      )}

      <div style={{ height: 16 }} />
      <BottomNav />
    </div>
  )
}
