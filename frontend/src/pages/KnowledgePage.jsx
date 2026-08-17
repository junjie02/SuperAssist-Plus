import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  AlertCircle,
  CheckCircle2,
  Clock3,
  FileText,
  LoaderCircle,
  RefreshCw,
  Trash2,
  UploadCloud,
  X,
} from 'lucide-react'
import { api } from '../lib/api'

const PENDING_STATUSES = new Set(['queued', 'parsing', 'indexing', 'deleting'])
const EMPTY_INDEX_STATUS = { stats: { documents: 0, chunks: 0 }, updated_at: '' }

export default function KnowledgePage() {
  const [documents, setDocuments] = useState([])
  const [indexStatus, setIndexStatus] = useState(EMPTY_INDEX_STATUS)
  const [selected, setSelected] = useState([])
  const [limits, setLimits] = useState({ max_file_size_mb: 25, max_files_per_batch: 20 })
  const [extensions, setExtensions] = useState([])
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [error, setError] = useState('')
  const inputRef = useRef(null)

  const loadDocuments = useCallback(async (quiet = false) => {
    if (!quiet) setLoading(true)
    try {
      const [data, indexData] = await Promise.all([
        api.get('/rag/documents'),
        api.get('/rag/graph'),
      ])
      setDocuments(data.documents || [])
      setLimits(data.limits || { max_file_size_mb: 25, max_files_per_batch: 20 })
      setExtensions(data.supported_extensions || [])
      setIndexStatus({ ...EMPTY_INDEX_STATUS, ...indexData })
      setError('')
    } catch (err) {
      setError(err.message || 'Unable to load the knowledge base.')
    } finally {
      if (!quiet) setLoading(false)
    }
  }, [])

  useEffect(() => { loadDocuments() }, [loadDocuments])

  const hasPending = documents.some(document => PENDING_STATUSES.has(document.status))
  useEffect(() => {
    if (!hasPending) return undefined
    const timer = window.setInterval(() => loadDocuments(true), 2500)
    return () => window.clearInterval(timer)
  }, [hasPending, loadDocuments])

  const accept = useMemo(() => extensions.join(','), [extensions])
  const addFiles = useCallback((fileList) => {
    const incoming = Array.from(fileList || [])
    setSelected(current => {
      const known = new Set(current.map(file => `${file.name}:${file.size}:${file.lastModified}`))
      const merged = [...current]
      for (const file of incoming) {
        const key = `${file.name}:${file.size}:${file.lastModified}`
        if (!known.has(key) && merged.length < limits.max_files_per_batch) {
          known.add(key)
          merged.push(file)
        }
      }
      return merged
    })
    setError('')
  }, [limits.max_files_per_batch])

  const upload = async () => {
    if (!selected.length || uploading) return
    setUploading(true)
    setError('')
    try {
      const form = new FormData()
      selected.forEach(file => form.append('files', file))
      const result = await api.post('/rag/documents', form)
      setSelected([])
      if (inputRef.current) inputRef.current.value = ''
      if (result.errors?.length) {
        setError(result.errors.map(item => `${item.name}: ${item.error}`).join('; '))
      }
      await loadDocuments(true)
    } catch (err) {
      setError(err.message || 'Upload failed.')
    } finally {
      setUploading(false)
    }
  }

  const removeDocument = async (document) => {
    if (PENDING_STATUSES.has(document.status)) return
    setError('')
    try {
      await api.del(`/rag/documents/${encodeURIComponent(document.id)}`)
      await loadDocuments(true)
    } catch (err) {
      setError(err.message || 'Delete failed.')
    }
  }

  return (
    <div className="knowledge-page">
      <header className="knowledge-header">
        <div>
          <h1>Knowledge Base</h1>
          <p>Hybrid document index</p>
        </div>
        <button className="icon-btn" onClick={() => loadDocuments()} disabled={loading} title="Refresh documents">
          <RefreshCw size={17} className={loading ? 'spin' : ''} aria-hidden="true" />
        </button>
      </header>

      <div className="knowledge-scroll">
        <div className="knowledge-content">
          <section
            className={`upload-tool ${dragging ? 'dragging' : ''}`}
            onDragEnter={(event) => { event.preventDefault(); setDragging(true) }}
            onDragOver={(event) => event.preventDefault()}
            onDragLeave={(event) => {
              if (!event.currentTarget.contains(event.relatedTarget)) setDragging(false)
            }}
            onDrop={(event) => {
              event.preventDefault()
              setDragging(false)
              addFiles(event.dataTransfer.files)
            }}
          >
            <UploadCloud size={28} aria-hidden="true" />
            <div className="upload-copy">
              <strong>Drop files here</strong>
              <span>{extensions.length ? extensions.join(', ') : 'TXT, PDF, Office, HTML, JSON and CSV'}</span>
            </div>
            <input
              ref={inputRef}
              type="file"
              multiple
              accept={accept}
              onChange={event => addFiles(event.target.files)}
              hidden
            />
            <button className="choose-files-btn" onClick={() => inputRef.current?.click()}>Choose files</button>
          </section>

          {selected.length > 0 && (
            <section className="upload-queue">
              <div className="upload-queue-head">
                <span>{selected.length} file{selected.length === 1 ? '' : 's'} selected</span>
                <button onClick={upload} disabled={uploading} className="upload-btn">
                  {uploading ? <LoaderCircle size={16} className="spin" /> : <UploadCloud size={16} />}
                  <span>{uploading ? 'Uploading' : 'Upload'}</span>
                </button>
              </div>
              <div className="selected-files">
                {selected.map((file, index) => (
                  <div className="selected-file" key={`${file.name}:${file.lastModified}`}>
                    <FileText size={15} aria-hidden="true" />
                    <span>{file.name}</span>
                    <small>{formatBytes(file.size)}</small>
                    <button onClick={() => setSelected(files => files.filter((_, i) => i !== index))} title="Remove file">
                      <X size={15} aria-hidden="true" />
                    </button>
                  </div>
                ))}
              </div>
            </section>
          )}

          {error && <div className="knowledge-error"><AlertCircle size={16} /> <span>{error}</span></div>}

          <section className="document-section">
            <div className="document-section-head">
              <h2>Documents</h2>
              <span>{documents.length} total</span>
            </div>
            {loading && documents.length === 0 ? (
              <div className="knowledge-empty"><LoaderCircle className="spin" size={22} /> Loading documents</div>
            ) : documents.length === 0 ? (
              <div className="knowledge-empty"><FileText size={24} /> No uploaded documents</div>
            ) : (
              <div className="document-list">
                {documents.map(document => (
                  <div className="document-row" key={document.id}>
                    <FileText size={20} aria-hidden="true" />
                    <div className="document-main">
                      <strong>{document.name}</strong>
                      <span>{formatBytes(document.size)} - {formatDate(document.created_at)}</span>
                      {document.error && <small>{document.error}</small>}
                    </div>
                    <Status status={document.status} />
                    <button
                      className="document-delete"
                      onClick={() => removeDocument(document)}
                      disabled={PENDING_STATUSES.has(document.status)}
                      title="Delete document"
                    >
                      <Trash2 size={17} aria-hidden="true" />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </section>

          <section className="knowledge-graph-section">
            <div className="knowledge-graph-head">
              <div>
                <h2>Search Index</h2>
                <span>{indexStatus.stats?.documents || 0} documents - {indexStatus.stats?.chunks || 0} original chunks</span>
              </div>
            </div>
            <div className="knowledge-empty">
              {hasPending ? <LoaderCircle className="spin" size={22} /> : <CheckCircle2 size={22} />}
              {hasPending
                ? 'Building the search index'
                : indexStatus.stats?.documents
                  ? 'Dense and BM25 indexes ready'
                  : 'Upload a document to build the search index'}
            </div>
          </section>
        </div>
      </div>
    </div>
  )
}

function Status({ status }) {
  const pending = PENDING_STATUSES.has(status)
  const failed = status === 'failed'
  const Icon = failed ? AlertCircle : pending ? Clock3 : CheckCircle2
  return (
    <span className={`document-status ${failed ? 'failed' : pending ? 'pending' : 'ready'}`}>
      <Icon size={14} />{status}
    </span>
  )
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return ''
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`
}

function formatDate(value) {
  if (!value) return ''
  try { return new Date(value).toLocaleString() } catch { return value }
}
