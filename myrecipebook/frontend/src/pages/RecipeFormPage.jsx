import { useState, useEffect } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { fetchRecipe, createRecipe, updateRecipe, uploadImage } from '../api/recipeApi'
import '../global.css'

const CATEGORIES     = ['和食', '洋食', '中華', 'イタリアン', 'アジアン', '副菜', 'その他']
const UNITS          = ['g', 'kg', 'ml', 'l', '大さじ', '小さじ', '個', '枚', '本', '適量']
const SERVING_OPTIONS= [0.5, 1, 2, 3, 4, 6]

const emptyIng  = () => ({ name: '', amount: '', unit: 'g' })
const emptyStep = (n) => ({ order: n, description: '', tip: '' })

export default function RecipeFormPage() {
  const navigate = useNavigate()
  const { id }   = useParams()
  const isEdit   = Boolean(id)

  const [form, setForm] = useState({
    title: '', category: '和食', description: '',
    base_servings: 2,
    prep_time: 10, cook_time: 30,
    ingredients: [emptyIng()],
    steps:       [emptyStep(1)],
  })
  const [imageFile, setImageFile] = useState(null)
  const [preview,   setPreview]   = useState(null)
  const [saving,    setSaving]    = useState(false)
  const [error,     setError]     = useState('')

  useEffect(() => {
    if (!isEdit) return
    fetchRecipe(id).then(r => {
      setForm({
        title:         r.title,
        category:      r.category,
        description:   r.description,
        base_servings: r.base_servings,
        prep_time:     r.prep_time,
        cook_time:     r.cook_time,
        ingredients:   r.ingredients.length ? r.ingredients : [emptyIng()],
        steps:         r.steps.length       ? r.steps.map(s => ({ ...s, tip: s.tip || '' })) : [emptyStep(1)],
      })
      if (r.image_url) setPreview(r.image_url)
    }).catch(() => setError('レシピの取得に失敗しました'))
  }, [id, isEdit])

  const set = (key, val) => setForm(f => ({ ...f, [key]: val }))

  const setIng = (i, key, val) =>
    setForm(f => {
      const arr = [...f.ingredients]; arr[i] = { ...arr[i], [key]: val }
      return { ...f, ingredients: arr }
    })
  const addIng    = ()  => setForm(f => ({ ...f, ingredients: [...f.ingredients, emptyIng()] }))
  const removeIng = (i) => setForm(f => ({ ...f, ingredients: f.ingredients.filter((_, j) => j !== i) }))

  const setStep = (i, key, val) =>
    setForm(f => {
      const arr = [...f.steps]; arr[i] = { ...arr[i], [key]: val }
      return { ...f, steps: arr }
    })
  const addStep    = ()  => setForm(f => ({
    ...f, steps: [...f.steps, emptyStep(f.steps.length + 1)]
  }))
  const removeStep = (i) => setForm(f => ({
    ...f, steps: f.steps.filter((_, j) => j !== i).map((s, idx) => ({ ...s, order: idx + 1 }))
  }))

  const handlePhoto = (e) => {
    const file = e.target.files[0]; if (!file) return
    setImageFile(file)
    setPreview(URL.createObjectURL(file))
  }

  const handleSubmit = async () => {
    if (!form.title.trim()) return setError('料理名を入力してください')
    setSaving(true); setError('')
    try {
      const payload = {
        ...form,
        base_servings: parseFloat(form.base_servings),
        prep_time:     parseInt(form.prep_time) || 0,
        cook_time:     parseInt(form.cook_time) || 0,
        ingredients: form.ingredients.filter(i => i.name.trim()).map(i => ({
          name: i.name, amount: parseFloat(i.amount) || 0, unit: i.unit,
        })),
        steps: form.steps.filter(s => s.description.trim()).map(s => ({
          order: s.order, description: s.description, tip: s.tip || null,
        })),
      }
      const saved = isEdit ? await updateRecipe(id, payload) : await createRecipe(payload)
      if (imageFile) await uploadImage(saved.id, imageFile)
      navigate(`/recipes/${saved.id}`)
    } catch {
      setError('保存に失敗しました。入力内容を確認してください。')
    } finally {
      setSaving(false)
    }
  }

  /* ── スタイルヘルパー ── */
  const S = {
    section: {
      background: 'var(--surface)', border: '1px solid var(--border)',
      borderRadius: 'var(--radius-md)', padding: '16px', marginBottom: 14,
    },
    sectionTitle: { fontSize: 14, fontWeight: 600, marginBottom: 12 },
    row2: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 },
    addLine: {
      display: 'flex', alignItems: 'center', gap: 6,
      fontSize: 13, color: 'var(--blue-500)', cursor: 'pointer', padding: '4px 0',
      background: 'none', border: 'none',
    },
    rmBtn: {
      width: 32, height: 32, background: '#fef2f2', color: '#b91c1c',
      border: 'none', borderRadius: 'var(--radius-sm)', cursor: 'pointer',
      fontSize: 16, flexShrink: 0,
    },
    stepN: {
      width: 28, height: 28, borderRadius: '50%',
      background: 'var(--blue-500)', color: '#fff',
      fontSize: 12, fontWeight: 600,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      flexShrink: 0, marginTop: 10,
    },
  }

  return (
    <div style={{ paddingBottom: 100 }}>
      {/* ── ヘッダー ── */}
      <div className="topbar">
        <div className="topbar-row">
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <button onClick={() => navigate(-1)}
              style={{ background: 'none', border: 'none', fontSize: 20, cursor: 'pointer', color: 'var(--text-2)' }}>
              <i className="ti ti-arrow-left" />
            </button>
            <span className="topbar-title">{isEdit ? 'レシピを編集' : '新しいレシピ'}</span>
          </div>
          <button className="btn btn-primary" onClick={handleSubmit} disabled={saving}>
            {saving ? '保存中…' : '保存'}
          </button>
        </div>
      </div>

      <div style={{ padding: '16px' }}>
        {error && <div className="error-banner">{error}</div>}

        {/* ── 基本情報 ── */}
        <div style={S.section}>
          <div style={S.sectionTitle}>基本情報</div>
          <div className="field">
            <label className="field-label">料理名 *</label>
            <input className="field-input" value={form.title}
              onChange={e => set('title', e.target.value)} placeholder="例：豚の角煮" />
          </div>
          <div className="field">
            <label className="field-label">カテゴリ</label>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
              {CATEGORIES.map(cat => (
                <button key={cat} onClick={() => set('category', cat)}
                  style={{
                    padding: '6px 14px', borderRadius: 999, fontSize: 13,
                    border: '1px solid',
                    borderColor: form.category === cat ? 'var(--blue-500)' : 'var(--border)',
                    background:  form.category === cat ? 'var(--blue-500)' : 'var(--surface)',
                    color:       form.category === cat ? '#fff' : 'var(--text-2)',
                    cursor: 'pointer',
                  }}>{cat}</button>
              ))}
            </div>
          </div>
          <div className="field">
            <label className="field-label">メモ・説明</label>
            <textarea className="field-input" value={form.description}
              onChange={e => set('description', e.target.value)}
              placeholder="料理の紹介や作るコツを一言で" />
          </div>
          <div style={S.row2}>
            <div className="field">
              <label className="field-label">基準の人数</label>
              <select className="field-input" value={form.base_servings}
                onChange={e => set('base_servings', parseFloat(e.target.value))}>
                {SERVING_OPTIONS.map(s => <option key={s} value={s}>{s}人前</option>)}
              </select>
            </div>
          </div>
          <div style={S.row2}>
            <div className="field">
              <label className="field-label">下準備（分）</label>
              <input className="field-input" type="number" min="0"
                value={form.prep_time} onChange={e => set('prep_time', e.target.value)} />
            </div>
            <div className="field">
              <label className="field-label">調理時間（分）</label>
              <input className="field-input" type="number" min="0"
                value={form.cook_time} onChange={e => set('cook_time', e.target.value)} />
            </div>
          </div>
        </div>

        {/* ── 写真 ── */}
        <div style={S.section}>
          <div style={S.sectionTitle}>料理写真（任意）</div>
          <label style={{
            display: 'block', border: '1.5px dashed var(--border)',
            borderRadius: 'var(--radius-md)', overflow: 'hidden', cursor: 'pointer',
          }}>
            {preview ? (
              <div style={{ position: 'relative' }}>
                <img src={preview} alt="プレビュー"
                  style={{ width: '100%', height: 180, objectFit: 'cover', display: 'block' }} />
                <div style={{
                  position: 'absolute', bottom: 8, right: 8,
                  background: 'rgba(0,0,0,.45)', color: '#fff',
                  padding: '4px 10px', borderRadius: 'var(--radius-sm)', fontSize: 12,
                }}>写真を変更</div>
              </div>
            ) : (
              <div style={{ padding: 24, textAlign: 'center', color: 'var(--text-3)' }}>
                <i className="ti ti-camera" style={{ fontSize: 30, display: 'block', marginBottom: 6 }} />
                <span style={{ fontSize: 13 }}>タップして写真を追加</span>
              </div>
            )}
            <input type="file" accept="image/*" style={{ display: 'none' }} onChange={handlePhoto} />
          </label>
        </div>

        {/* ── 材料 ── */}
        <div style={S.section}>
          <div style={S.sectionTitle}>材料</div>
          {form.ingredients.map((ing, i) => (
            <div key={i} style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 8 }}>
              <input className="field-input" placeholder="食材名"
                value={ing.name} onChange={e => setIng(i, 'name', e.target.value)}
                style={{ flex: 1 }} />
              <input className="field-input" placeholder="量"
                value={ing.amount} onChange={e => setIng(i, 'amount', e.target.value)}
                style={{ width: 62, flex: 'none' }} />
              <select className="field-input" value={ing.unit}
                onChange={e => setIng(i, 'unit', e.target.value)}
                style={{ width: 72, flex: 'none', paddingLeft: 6 }}>
                {UNITS.map(u => <option key={u} value={u}>{u}</option>)}
              </select>
              <button style={S.rmBtn} onClick={() => removeIng(i)}
                disabled={form.ingredients.length === 1}>
                <i className="ti ti-minus" />
              </button>
            </div>
          ))}
          <button style={S.addLine} onClick={addIng}>
            <i className="ti ti-plus" style={{ fontSize: 16 }} />材料を追加
          </button>
        </div>

        {/* ── 手順 ── */}
        <div style={S.section}>
          <div style={S.sectionTitle}>作り方の手順</div>
          {form.steps.map((step, i) => (
            <div key={i} style={{ display: 'flex', gap: 10, alignItems: 'flex-start', marginBottom: 12 }}>
              <div style={S.stepN}>{i + 1}</div>
              <div style={{ flex: 1 }}>
                <textarea className="field-input"
                  placeholder={`工程 ${i + 1} の説明`}
                  value={step.description}
                  onChange={e => setStep(i, 'description', e.target.value)}
                  style={{ minHeight: 68, marginBottom: 6 }} />
                <input className="field-input"
                  placeholder="💡 ポイント・ヒント（任意）"
                  value={step.tip}
                  onChange={e => setStep(i, 'tip', e.target.value)} />
              </div>
              <button style={{ ...S.rmBtn, marginTop: 10 }} onClick={() => removeStep(i)}
                disabled={form.steps.length === 1}>
                <i className="ti ti-minus" />
              </button>
            </div>
          ))}
          <button style={S.addLine} onClick={addStep}>
            <i className="ti ti-plus" style={{ fontSize: 16 }} />工程を追加
          </button>
        </div>

        {/* ── 保存・キャンセル ── */}
        <div style={{ display: 'flex', gap: 10 }}>
          <button className="btn btn-ghost" style={{ flex: 1 }} onClick={() => navigate(-1)}>
            キャンセル
          </button>
          <button className="btn btn-primary" style={{ flex: 2 }} onClick={handleSubmit} disabled={saving}>
            {saving ? '保存中…' : isEdit ? '変更を保存' : 'レシピを登録'}
          </button>
        </div>
      </div>
    </div>
  )
}
