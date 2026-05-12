import { Link, useLocation } from 'react-router-dom'

export default function BottomNav() {
  const { pathname } = useLocation()

  const items = [
    { to: '/recipes',   icon: 'ti-home',        label: 'ホーム'     },
    { to: '/favorites', icon: 'ti-heart',        label: 'お気に入り' },
    { to: '/recipes/new', icon: 'ti-plus-circle', label: '追加'       },
  ]

  return (
    <nav className="bnav">
      {items.map(item => (
        <Link
          key={item.to}
          to={item.to}
          className={`bnav-item ${pathname === item.to ? 'active' : ''}`}
        >
          <i className={`ti ${item.icon}`} aria-hidden="true" />
          {item.label}
        </Link>
      ))}
    </nav>
  )
}
