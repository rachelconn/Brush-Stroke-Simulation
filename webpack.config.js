const path = require('path');

module.exports = {
  devtool: 'source-map',
  entry: './src/index.ts',
  module: {
    rules: [
      // CSS modules
      {
        test: /\.css$/,
        exclude: /node_modules/,
        use: [
          'style-loader',
          {
            loader: '@teamsupercell/typings-for-css-modules-loader',
          },
          {
            loader: 'css-loader',
            options: {
              modules: {
                mode: 'local',
                localIdentName: '[local]--[hash:base64:5]',
                exportLocalsConvention: 'camelCase',
              },
            },
          },
        ],
      },
      // TypeScript compilation
      {
        test: /\.tsx?$/,
        use: 'ts-loader',
        exclude: /node_modules/,
      },
      // Asset modules for loading files: see https://webpack.js.org/guides/asset-modules/
      {
        test: /\.(png|jpg)$/i,
        type: 'asset/resource',
      },
      // Serve svgs as base64-encoded strings
      {
        test: /\.(svg)$/i,
        type: 'asset/inline',
      },
    ],
  },
  resolve: {
    extensions: ['.tsx', '.ts', '.js', '.css'],
  },
  output: {
    filename: 'main.js',
    path: path.resolve(__dirname, '.'),
    assetModuleFilename: 'static/[hash][ext][query]',
  },
};
