def get_custom_css():
    return """
    <style>
        /* General Page Styling */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        .main-header {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0369a1 100%);
            padding: 24px 30px;
            border-radius: 14px;
            color: #ffffff;
            margin-bottom: 24px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
        }
        
        .main-header h1 {
            font-size: 2.1rem;
            font-weight: 700;
            color: #ffffff;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .main-header p {
            font-size: 0.98rem;
            color: #94a3b8;
            margin-top: 8px;
            margin-bottom: 0;
        }

        /* Stat Cards */
        .stat-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 18px;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .stat-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        }

        .stat-val {
            font-size: 1.8rem;
            font-weight: 700;
            color: #0284c7;
        }

        .stat-lbl {
            font-size: 0.85rem;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 4px;
        }

        /* Citation Card */
        .citation-card {
            background: #f8fafc;
            border-left: 4px solid #0284c7;
            border-radius: 0 8px 8px 0;
            padding: 14px 16px;
            margin-bottom: 12px;
            border-top: 1px solid #e2e8f0;
            border-right: 1px solid #e2e8f0;
            border-bottom: 1px solid #e2e8f0;
        }

        .citation-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
        }

        .citation-title {
            font-weight: 600;
            font-size: 0.95rem;
            color: #0f172a;
        }

        .citation-badge {
            background: #e0f2fe;
            color: #0369a1;
            padding: 2px 8px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
        }

        .citation-quote {
            font-size: 0.88rem;
            color: #334155;
            font-style: italic;
            line-height: 1.45;
            background: #ffffff;
            padding: 8px 12px;
            border-radius: 6px;
            border: 1px solid #f1f5f9;
        }

        /* Evaluation Score Card */
        .eval-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 16px;
        }

        .badge-low-risk {
            background-color: #dcfce7;
            color: #166534;
            padding: 4px 12px;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.85rem;
        }

        .badge-mod-risk {
            background-color: #fef9c3;
            color: #854d0e;
            padding: 4px 12px;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.85rem;
        }

        .badge-high-risk {
            background-color: #fee2e2;
            color: #991b1b;
            padding: 4px 12px;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.85rem;
        }
    </style>
    """
