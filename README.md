# Bias Testing App

An open source React application for testing AI systems for bias and disparate impact across protected groups. Powered by the Anthropic Claude API.

## What it does

This tool helps teams evaluate AI systems for fairness by analyzing outputs against common bias metrics, including:

- Demographic parity (equal selection rates across groups)
- Equalized odds (equal true positive and false positive rates)
- Predictive parity (equal precision across groups)
- Disparate impact ratio (four-fifths rule)

Upload test data, specify protected groups, and the app uses Claude to generate a structured bias testing report.

## Requirements

- Node.js 18 or later
- npm (comes with Node.js)
- An Anthropic API key (get one at [console.anthropic.com](https://console.anthropic.com))

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR-USERNAME/bias-testing-app.git
   cd bias-testing-app
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Create your environment file:
   ```bash
   cp .env.example .env
   ```

4. Open `.env` and add your Anthropic API key:
   ```
   VITE_ANTHROPIC_API_KEY=sk-ant-...
   ```

5. Run the development server:
   ```bash
   npm run dev
   ```

   The app opens at `http://localhost:5173`.

## Available Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Start the development server with hot reload |
| `npm run build` | Build for production (outputs to `dist/`) |
| `npm run preview` | Preview the production build locally |
| `npm run lint` | Run ESLint |

## Deployment

### Static hosting (Vercel, Netlify, Azure Static Web Apps, GitHub Pages)

```bash
npm run build
```

The `dist/` directory contains the built app ready to deploy to any static host. Remember to configure `VITE_ANTHROPIC_API_KEY` as an environment variable in your hosting platform, not in the repo.

### Self-hosted

```bash
npm run build
npm run preview
```

## Security Notes

- **Never commit `.env` to git.** The `.gitignore` is configured to exclude it, but always verify before committing.
- This build exposes the Anthropic API key to the browser. For public-facing deployments, proxy API calls through a backend service so the key stays server-side.
- If you are using this tool with sensitive or regulated data, make sure your deployment environment meets the applicable compliance requirements.

## Project Structure

```
bias-testing-app/
├── .env.example           # Template for environment variables
├── .gitignore             # Excludes node_modules, .env, build output
├── .eslintrc.cjs          # ESLint configuration
├── LICENSE                # MIT license
├── README.md              # This file
├── index.html             # Vite entry HTML
├── package.json           # Dependencies and scripts
├── postcss.config.js      # PostCSS config for Tailwind
├── tailwind.config.js     # Tailwind CSS config
├── vite.config.js         # Vite build config
└── src/
    ├── App.jsx            # Main component
    ├── main.jsx           # React entry point
    └── index.css          # Global styles and Tailwind directives
```

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to open a pull request or file an issue.

## License

MIT License. See [LICENSE](LICENSE) for details.
