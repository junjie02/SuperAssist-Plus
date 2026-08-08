import { useCallback, useEffect, useState } from 'react'
import {
  CheckCircle2,
  Eye,
  EyeOff,
  LoaderCircle,
  RotateCcw,
  Save,
  Trash2,
} from 'lucide-react'
import { api } from '../lib/api'
import { useAuth } from '../hooks/useAuth'

const MEMORY_SECTIONS = [
  {
    title: 'Write path',
    fields: [
      { key: 'llm_writer_enabled', label: 'LLM writer', env: 'SUPERASSIST_MEMORY_LLM_WRITER_ENABLED', type: 'boolean' },
      { key: 'debounce_seconds', label: 'Write debounce', env: 'SUPERASSIST_MEMORY_DEBOUNCE_SECONDS', suffix: 'seconds', min: 0, max: 3600, step: 0.1 },
    ],
  },
  {
    title: 'Read path',
    fields: [
      { key: 'top_k', label: 'Injected nodes', env: 'SUPERASSIST_MEMORY_TOP_K', min: 1, max: 1000, step: 1 },
      { key: 'candidate_pool_size', label: 'Candidate pool', env: 'SUPERASSIST_MEMORY_CANDIDATE_POOL_SIZE', min: 1, max: 10000, step: 1 },
      { key: 'read_use_ppr', label: 'Use Personalized PageRank', env: 'SUPERASSIST_MEMORY_READ_USE_PPR', type: 'boolean' },
      { key: 'read_entry_points', label: 'Vector entry points', env: 'SUPERASSIST_MEMORY_READ_ENTRY_POINTS', min: 1, max: 1000, step: 1 },
      { key: 'read_max_depth', label: 'BFS max depth', env: 'SUPERASSIST_MEMORY_READ_MAX_DEPTH', min: 1, max: 20, step: 1 },
      { key: 'read_bfs_weight', label: 'BFS weight', env: 'SUPERASSIST_MEMORY_READ_BFS_WEIGHT', min: 0, max: 1, step: 0.05 },
      { key: 'read_ppr_weight', label: 'PPR weight', env: 'SUPERASSIST_MEMORY_READ_PPR_WEIGHT', min: 0, max: 1, step: 0.05 },
      { key: 'read_bfs_decay', label: 'BFS hop decay', env: 'SUPERASSIST_MEMORY_READ_BFS_DECAY', min: 0, max: 1, step: 0.05 },
    ],
  },
  {
    title: 'Consolidation and decay',
    fields: [
      { key: 'reinforce_similarity', label: 'Reinforce similarity', env: 'SUPERASSIST_MEMORY_REINFORCE_SIMILARITY', min: 0, max: 1, step: 0.01 },
      { key: 'concept_merge_similarity', label: 'Concept merge similarity', env: 'SUPERASSIST_MEMORY_CONCEPT_MERGE_SIMILARITY', min: 0, max: 1, step: 0.01 },
      { key: 'completion_similarity', label: 'Completion similarity', env: 'SUPERASSIST_MEMORY_COMPLETION_SIMILARITY', min: 0, max: 1, step: 0.01 },
      { key: 'completion_top_k', label: 'Completion candidates', env: 'SUPERASSIST_MEMORY_COMPLETION_TOP_K', min: 1, max: 1000, step: 1 },
      { key: 'decay_lambda', label: 'Decay lambda', env: 'SUPERASSIST_MEMORY_DECAY_LAMBDA', min: 0, max: 10, step: 0.001 },
      { key: 'edge_delete_threshold', label: 'Edge delete threshold', env: 'SUPERASSIST_MEMORY_EDGE_DELETE_THRESHOLD', min: 0, max: 1, step: 0.01 },
    ],
  },
  {
    title: 'Short memory',
    fields: [
      { key: 'short_token_limit', label: 'Compression token limit', env: 'SUPERASSIST_SHORT_MEMORY_TOKEN_LIMIT', min: 100, max: 10000000, step: 100 },
      { key: 'short_keep_recent_turns', label: 'Sliding window turns', env: 'SUPERASSIST_SHORT_MEMORY_KEEP_RECENT_TURNS', min: 1, max: 10000, step: 1 },
      { key: 'short_summary_target_tokens', label: 'Summary target tokens', env: 'SUPERASSIST_SHORT_MEMORY_SUMMARY_TARGET_TOKENS', min: 1, max: 1000000, step: 100 },
    ],
  },
]

export default function SettingsPage() {
  const { user } = useAuth()
  const [activeTab, setActiveTab] = useState('memory')
  const [draft, setDraft] = useState(null)
  const [secret, setSecret] = useState('')
  const [secretDirty, setSecretDirty] = useState(false)
  const [showSecret, setShowSecret] = useState(false)
  const [wecomSecret, setWecomSecret] = useState('')
  const [wecomSecretDirty, setWecomSecretDirty] = useState(false)
  const [showWecomSecret, setShowWecomSecret] = useState(false)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [saved, setSaved] = useState(false)
  const [restartRequired, setRestartRequired] = useState(false)

  const load = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const data = await api.get('/settings')
      setDraft({ memory: data.memory, skills: data.skills, feishu: data.feishu, wecom: data.wecom })
      setSecret('')
      setSecretDirty(false)
      setWecomSecret('')
      setWecomSecretDirty(false)
      setRestartRequired(false)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { load() }, [load])

  const updateMemory = (key, value) => {
    setSaved(false)
    setDraft(current => ({ ...current, memory: { ...current.memory, [key]: value } }))
  }

  const updateFeishu = (key, value) => {
    setSaved(false)
    setDraft(current => ({ ...current, feishu: { ...current.feishu, [key]: value } }))
  }

  const updateSkills = (key, value) => {
    setSaved(false)
    setDraft(current => ({ ...current, skills: { ...current.skills, [key]: value } }))
  }

  const updateWeCom = (key, value) => {
    setSaved(false)
    setDraft(current => ({ ...current, wecom: { ...current.wecom, [key]: value } }))
  }

  const handleSubmit = async (event) => {
    event.preventDefault()
    setSaving(true)
    setError('')
    setSaved(false)
    const feishu = {
      app_id: draft.feishu.app_id,
      domain: draft.feishu.domain,
      allowed_open_ids: draft.feishu.allowed_open_ids,
      mention_only: draft.feishu.mention_only,
      active_session_seconds: draft.feishu.active_session_seconds,
    }
    if (secretDirty) feishu.app_secret = secret
    const wecom = {
      bot_id: draft.wecom.bot_id,
      allowed_user_ids: draft.wecom.allowed_user_ids,
      user_id_map: draft.wecom.user_id_map,
      rag_mode_default: draft.wecom.rag_mode_default,
      max_concurrent: draft.wecom.max_concurrent,
      stream_interval_ms: draft.wecom.stream_interval_ms,
      ai_engine_url: draft.wecom.ai_engine_url,
      rpa_allowed_groups: draft.wecom.rpa_allowed_groups,
      rpa_trigger_prefixes: draft.wecom.rpa_trigger_prefixes,
      rpa_poll_interval_seconds: draft.wecom.rpa_poll_interval_seconds,
      rpa_reply_max_chars: draft.wecom.rpa_reply_max_chars,
    }
    if (wecomSecretDirty) wecom.bot_secret = wecomSecret

    try {
      const data = await api.put('/settings', { memory: draft.memory, skills: draft.skills, feishu, wecom })
      setDraft({ memory: data.memory, skills: data.skills, feishu: data.feishu, wecom: data.wecom })
      setSecret('')
      setSecretDirty(false)
      setWecomSecret('')
      setWecomSecretDirty(false)
      setRestartRequired(data.meta?.feishu_restart_required === true || data.meta?.wecom_restart_required === true)
      setSaved(true)
    } catch (e) {
      setError(e.message)
    } finally {
      setSaving(false)
    }
  }

  if (loading) {
    return <div className="settings-page settings-state"><LoaderCircle className="spin" size={22} /> Loading settings...</div>
  }

  if (!draft) {
    return (
      <div className="settings-page settings-state error">
        <span>{error || 'Settings are unavailable.'}</span>
        <button className="icon-btn" onClick={load} title="Retry"><RotateCcw size={18} /></button>
      </div>
    )
  }

  return (
    <form className="settings-page" onSubmit={handleSubmit}>
      <header className="settings-header">
        <div>
          <h1>Settings</h1>
          <p>Runtime configuration</p>
        </div>
        <div className="settings-actions">
          <button type="button" className="icon-btn" onClick={load} title="Reload settings" disabled={saving}>
            <RotateCcw size={18} />
          </button>
          <button type="submit" className="save-settings-btn" disabled={saving}>
            {saving ? <LoaderCircle className="spin" size={17} /> : <Save size={17} />}
            <span>{saving ? 'Saving' : 'Save changes'}</span>
          </button>
        </div>
      </header>

      <div className="settings-tabs" role="tablist" aria-label="Settings sections">
        <button type="button" role="tab" aria-selected={activeTab === 'memory'} className={activeTab === 'memory' ? 'active' : ''} onClick={() => setActiveTab('memory')}>Memory</button>
        <button type="button" role="tab" aria-selected={activeTab === 'skills'} className={activeTab === 'skills' ? 'active' : ''} onClick={() => setActiveTab('skills')}>Skills</button>
        <button type="button" role="tab" aria-selected={activeTab === 'feishu'} className={activeTab === 'feishu' ? 'active' : ''} onClick={() => setActiveTab('feishu')}>Feishu</button>
        <button type="button" role="tab" aria-selected={activeTab === 'wecom'} className={activeTab === 'wecom' ? 'active' : ''} onClick={() => setActiveTab('wecom')}>WeCom</button>
      </div>

      <div className="settings-scroll">
        <div className="settings-content">
          {error && <div className="settings-notice error-notice">{error}</div>}
          {saved && (
            <div className="settings-notice success-notice">
              <CheckCircle2 size={18} />
              <span>{restartRequired ? 'Saved. Restart the affected channel process to apply connection changes.' : 'Saved and applied to new requests.'}</span>
            </div>
          )}

          {activeTab === 'memory' ? (
            MEMORY_SECTIONS.map(section => (
              <section className="settings-section" key={section.title}>
                <h2>{section.title}</h2>
                <div className="settings-grid">
                  {section.fields.map(field => (
                    field.type === 'boolean' ? (
                      <ToggleField
                        key={field.key}
                        label={field.label}
                        env={field.env}
                        checked={Boolean(draft.memory[field.key])}
                        onChange={value => updateMemory(field.key, value)}
                      />
                    ) : (
                      <NumberField
                        key={field.key}
                        {...field}
                        value={draft.memory[field.key]}
                        onChange={value => updateMemory(field.key, value)}
                      />
                    )
                  ))}
                </div>
              </section>
            ))
          ) : activeTab === 'skills' ? (
            <section className="settings-section skills-section">
              <h2>Progressive loading</h2>
              <div className="settings-grid">
                <NumberField label="Active skill timeout" env="SUPERASSIST_SKILL_ACTIVE_TTL_SECONDS" value={draft.skills.active_ttl_seconds} onChange={value => updateSkills('active_ttl_seconds', value)} suffix="seconds" min={30} max={86400} step={30} />
              </div>
            </section>
          ) : activeTab === 'feishu' ? (
            <section className="settings-section feishu-section">
              <h2>Connection</h2>
              <div className="settings-grid">
                <TextField label="App ID" env="SUPERASSIST_FEISHU_APP_ID" value={draft.feishu.app_id} onChange={value => updateFeishu('app_id', value)} autoComplete="off" />
                <label className="settings-field">
                  <span className="field-label">App Secret</span>
                  <span className="field-env">SUPERASSIST_FEISHU_APP_SECRET</span>
                  <span className="secret-input-wrap">
                    <input
                      type={showSecret ? 'text' : 'password'}
                      value={secret}
                      onChange={event => { setSecret(event.target.value); setSecretDirty(true); setSaved(false) }}
                      placeholder={draft.feishu.app_secret_configured && !secretDirty ? 'Configured - leave blank to keep' : 'Not configured'}
                      autoComplete="new-password"
                    />
                    <button type="button" className="field-icon-btn" onClick={() => setShowSecret(value => !value)} title={showSecret ? 'Hide secret' : 'Show secret'} disabled={!secret}>
                      {showSecret ? <EyeOff size={17} /> : <Eye size={17} />}
                    </button>
                  </span>
                  {draft.feishu.app_secret_configured && !secretDirty && (
                    <button type="button" className="clear-secret-btn" onClick={() => { setSecret(''); setSecretDirty(true); setSaved(false) }}>
                      <Trash2 size={14} /> Clear configured secret
                    </button>
                  )}
                  {secretDirty && secret === '' && <span className="field-warning">Secret will be cleared when saved.</span>}
                </label>
                <TextField label="Open API domain" env="SUPERASSIST_FEISHU_DOMAIN" type="url" value={draft.feishu.domain} onChange={value => updateFeishu('domain', value)} required />
                <TextField label="Allowed Open IDs" env="SUPERASSIST_FEISHU_ALLOWED_OPEN_IDS" value={draft.feishu.allowed_open_ids} onChange={value => updateFeishu('allowed_open_ids', value)} placeholder="ou_xxx,ou_yyy" />
                <ToggleField label="Mention to start session" env="SUPERASSIST_FEISHU_MENTION_ONLY" checked={Boolean(draft.feishu.mention_only)} onChange={value => updateFeishu('mention_only', value)} />
                <NumberField label="Batch quiet window" env="SUPERASSIST_FEISHU_ACTIVATION_DEBOUNCE_SECONDS" value={draft.feishu.activation_debounce_seconds} onChange={value => updateFeishu('activation_debounce_seconds', value)} suffix="seconds" min={0} max={30} step={0.1} />
                <NumberField label="Batch maximum wait" env="SUPERASSIST_FEISHU_ACTIVATION_MAX_WAIT_SECONDS" value={draft.feishu.activation_max_wait_seconds} onChange={value => updateFeishu('activation_max_wait_seconds', value)} suffix="seconds" min={0.1} max={60} step={0.5} />
                <NumberField label="Images per activation" env="SUPERASSIST_FEISHU_MAX_IMAGES_PER_ACTIVATION" value={draft.feishu.max_images_per_activation} onChange={value => updateFeishu('max_images_per_activation', value)} min={1} max={100} step={1} />
              </div>
            </section>
          ) : (
            <>
              <section className="settings-section wecom-section">
                <h2>Intelligent robot connection</h2>
                <div className="settings-grid">
                <TextField label="Bot ID" env="SUPERASSIST_WECOM_BOT_ID" value={draft.wecom.bot_id} onChange={value => updateWeCom('bot_id', value)} autoComplete="off" />
                <label className="settings-field">
                  <span className="field-label">Bot Secret</span>
                  <span className="field-env">SUPERASSIST_WECOM_BOT_SECRET</span>
                  <span className="secret-input-wrap">
                    <input
                      type={showWecomSecret ? 'text' : 'password'}
                      value={wecomSecret}
                      onChange={event => { setWecomSecret(event.target.value); setWecomSecretDirty(true); setSaved(false) }}
                      placeholder={draft.wecom.bot_secret_configured && !wecomSecretDirty ? 'Configured - leave blank to keep' : 'Not configured'}
                      autoComplete="new-password"
                    />
                    <button type="button" className="field-icon-btn" onClick={() => setShowWecomSecret(value => !value)} title={showWecomSecret ? 'Hide secret' : 'Show secret'} disabled={!wecomSecret}>
                      {showWecomSecret ? <EyeOff size={17} /> : <Eye size={17} />}
                    </button>
                  </span>
                  {draft.wecom.bot_secret_configured && !wecomSecretDirty && (
                    <button type="button" className="clear-secret-btn" onClick={() => { setWecomSecret(''); setWecomSecretDirty(true); setSaved(false) }}>
                      <Trash2 size={14} /> Clear configured secret
                    </button>
                  )}
                  {wecomSecretDirty && wecomSecret === '' && <span className="field-warning">Secret will be cleared when saved.</span>}
                </label>
                <TextField label="AI Engine URL" env="SUPERASSIST_WECOM_AI_ENGINE_URL" type="url" value={draft.wecom.ai_engine_url} onChange={value => updateWeCom('ai_engine_url', value)} required />
                <TextField label="Allowed user IDs" env="SUPERASSIST_WECOM_ALLOWED_USER_IDS" value={draft.wecom.allowed_user_ids} onChange={value => updateWeCom('allowed_user_ids', value)} placeholder="zhangsan,lisi" />
                <TextField label="Current SuperAssist user ID" value={user?.id || ''} onChange={() => {}} readOnly />
                <TextField label="WeCom to SuperAssist user map" env="SUPERASSIST_WECOM_USER_ID_MAP" value={draft.wecom.user_id_map} onChange={value => updateWeCom('user_id_map', value)} placeholder={'{"zhangsan":"user_xxx","chat:group_id":"user_shared"}'} spellCheck="false" />
                <NumberField label="Maximum concurrent requests" env="SUPERASSIST_WECOM_MAX_CONCURRENT" value={draft.wecom.max_concurrent} onChange={value => updateWeCom('max_concurrent', value)} min={1} max={32} step={1} />
                <NumberField label="Stream update interval" env="SUPERASSIST_WECOM_STREAM_INTERVAL_MS" value={draft.wecom.stream_interval_ms} onChange={value => updateWeCom('stream_interval_ms', value)} suffix="ms" min={100} max={5000} step={50} />
                <ToggleField label="Enable RAG by default" env="SUPERASSIST_WECOM_RAG_MODE_DEFAULT" checked={Boolean(draft.wecom.rag_mode_default)} onChange={value => updateWeCom('rag_mode_default', value)} />
                </div>
              </section>
              <section className="settings-section wecom-section">
                <h2>Desktop RPA for external groups</h2>
                <div className="settings-grid">
                  <TextField label="Allowed group names" env="SUPERASSIST_WECOM_RPA_ALLOWED_GROUPS" value={draft.wecom.rpa_allowed_groups} onChange={value => updateWeCom('rpa_allowed_groups', value)} placeholder="项目答疑群,客户交流群" />
                  <TextField label="Wake prefixes" env="SUPERASSIST_WECOM_RPA_TRIGGER_PREFIXES" value={draft.wecom.rpa_trigger_prefixes} onChange={value => updateWeCom('rpa_trigger_prefixes', value)} placeholder="@SuperAssist,小助手" />
                  <NumberField label="Polling interval" env="SUPERASSIST_WECOM_RPA_POLL_INTERVAL_SECONDS" value={draft.wecom.rpa_poll_interval_seconds} onChange={value => updateWeCom('rpa_poll_interval_seconds', value)} suffix="seconds" min={0.5} max={30} step={0.5} />
                  <NumberField label="Reply chunk size" env="SUPERASSIST_WECOM_RPA_REPLY_MAX_CHARS" value={draft.wecom.rpa_reply_max_chars} onChange={value => updateWeCom('rpa_reply_max_chars', value)} suffix="characters" min={100} max={10000} step={100} />
                </div>
              </section>
            </>
          )}
        </div>
      </div>
    </form>
  )
}

function NumberField({ label, env, value, onChange, suffix, min, max, step }) {
  return (
    <label className="settings-field">
      <span className="field-label">{label}</span>
      <span className="field-env">{env}</span>
      <span className="number-input-wrap">
        <input type="number" value={value} min={min} max={max} step={step} required onChange={event => onChange(event.target.value)} />
        {suffix && <span>{suffix}</span>}
      </span>
    </label>
  )
}

function TextField({ label, env, value, onChange, ...inputProps }) {
  return (
    <label className="settings-field">
      <span className="field-label">{label}</span>
      <span className="field-env">{env}</span>
      <input value={value} onChange={event => onChange(event.target.value)} {...inputProps} />
    </label>
  )
}

function ToggleField({ label, env, checked, onChange }) {
  return (
    <div className="settings-field toggle-field">
      <span>
        <span className="field-label">{label}</span>
        <span className="field-env">{env}</span>
      </span>
      <button type="button" className={`switch ${checked ? 'on' : ''}`} role="switch" aria-checked={checked} onClick={() => onChange(!checked)}>
        <span />
      </button>
    </div>
  )
}
