FROM node:18-alpine

WORKDIR /app

# Copy package.json files
COPY package.json pnpm-lock.yaml ./

# Install dependencies using npm (to avoid pnpm version issues)
RUN npm install --legacy-peer-deps --no-fund --no-audit

# Copy application code
COPY . .

# Build the Next.js app
RUN npm run build

# Expose port
EXPOSE 3000

# Run the application
CMD ["npm", "start"]