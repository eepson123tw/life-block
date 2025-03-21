# 生命時間線應用（Life Timeline）規格書

## 1. 專案概述

### 1.1 專案名稱
生命時間線（Life Timeline）

### 1.2 願景
創建一個直觀的可視化工具，幫助用戶以週、月、日、年等不同時間單位理解自己的生命歷程，結合 LLM 智能生成的詩意祝福，促使人們珍惜時間和思考生命意義。

### 1.3 目標用戶
- 對生命規劃有興趣的個人
- 尋求自我反思和生命意義的人
- 喜歡數據可視化和個人分析的用戶

## 2. 技術架構

### 2.1 前端
- **框架**：React + Vite
- **樣式**：Tailwind CSS
- **視覺化**：Custom React Components
- **數據獲取**：Axios/React Query
- **部署**：Vercel/Netlify

### 2.2 後端
- **框架**：Python + FastAPI
- **AI 整合**：Anthropic Claude API
- **資料搜索**：Web Search API
- **部署**：Railway/Render

### 2.3 資料存儲
初始版本不需資料庫，使用：
- 瀏覽器本地存儲（LocalStorage）- 保存用戶設定
- 靜態 JSON 資料 - 存儲人口統計學資料

## 3. 功能規格

### 3.1 用戶輸入

#### 3.1.1 必要輸入
- 出生日期（年月日）
- 性別（男/女）

#### 3.1.2 可選輸入
- 國家/地區
- 自訂里程碑事件

### 3.2 時間線視覺化

#### 3.2.1 時間單位切換
- 天視圖：每行 30 天，代表大約一個月
- 週視圖：每行 52 週，代表一年
- 月視圖：每行 12 個月，代表一年
- 年視圖：每行 10 年，代表一個十年

#### 3.2.2 生命階段顯示
- 以不同顏色區分生命階段：
  - 幼年期（0-5歲）
  - 小學階段（6-10歲）
  - 中學階段（11-14歲）
  - 高中階段（15-18歲）
  - 大學階段（19-22歲）
  - 研究生階段（23-25歲）
  - 職業生涯（26-65歲）
  - 退休期（66歲以上）

#### 3.2.3 視覺元素
- 區分已過和未來時間（不同透明度）
- 右側生命階段顏色說明
- 當前時間點標記（可選）

### 3.3 智能生成內容

#### 3.3.1 平均壽命計算
- 基於用戶的國家和性別獲取準確平均壽命
- 使用 LLM 工具搜索最新人口統計學數據

#### 3.3.2 詩意祝福生成
- 基於用戶當前生命階段生成個性化祝福
- 考慮文化背景和價值觀
- 提供哲理性思考和鼓勵

### 3.4 互動功能

#### 3.4.1 基本互動
- 時間單位切換
- 格子懸停顯示詳細信息

#### 3.4.2 自訂功能（V2版本）
- 添加自定義里程碑
- 調整生命階段
- 自訂顏色方案

## 4. 用戶界面設計

### 4.1 主頁結構
```
+-----------------------------------------------+
|  NavBar: Logo + 關於 + 幫助                    |
+-----------------------------------------------+
|                                               |
|  主標題: Life in Perspective                   |
|  副標題: 重新認識你生命中的每一個時刻            |
|                                               |
|  [視覺化圖像或動畫]                            |
|                                               |
|  簡短的引言文字                                |
|                                               |
+-----------------------------------------------+
|                                               |
|  輸入區塊                                      |
|  +-----------------------------------------+  |
|  |  出生日期 | 性別選擇 | 國家/語言(可選)    |  |
|  |  [   開始我的生命時間線   ]               |  |
|  +-----------------------------------------+  |
|                                               |
+-----------------------------------------------+
```

### 4.2 時間線頁面結構
```
+-----------------------------------------------+
|  NavBar: Logo + 返回主頁 + 分享               |
+-----------------------------------------------+
|                                               |
|  用戶資訊: 出生日期 | 性別 | 國家 | 當前年齡   |
|                                               |
|  [時間單位切換按鈕]                           |
|                                               |
|  +------------------+    +------------------+ |
|  |                  |    |                  | |
|  |                  |    |  生命階段說明     | |
|  |  時間線視覺化網格  |    |  顏色區塊及文字   | |
|  |                  |    |                  | |
|  |                  |    |                  | |
|  +------------------+    +------------------+ |
|                                               |
|  生命統計數據: 已過時間 | 剩餘時間 | 百分比     |
|                                               |
|  +------------------+                         |
|  |  AI 生成的詩意祝福                        | |
|  |  和對當前生命階段的思考                    | |
|  +------------------+                         |
|                                               |
+-----------------------------------------------+
|  頁尾: 隱私聲明 | 關於                        |
+-----------------------------------------------+
```

## 5. AI 生成流程

### 5.1 壽命數據獲取流程
1. 接收用戶輸入（出生日期、性別、國家）
2. 如未提供國家，使用 LLM 推測可能的國家
3. 構建搜索查詢（如 "average life expectancy for male in Taiwan latest data"）
4. 使用 Web Search 工具執行搜索
5. 使用 LLM 解析搜索結果，提取平均壽命數據
6. 返回準確平均壽命數值

### 5.2 詩意祝福生成流程
1. 接收用戶數據和生命統計資訊
2. 確定用戶當前生命階段
3. 構建 LLM 提示詞，包含：
   - 用戶年齡、性別和國家
   - 已度過生命的百分比
   - 當前生命階段特點
   - 文化和哲學思考要素
4. 使用 Claude API 生成詩意祝福
5. 返回生成的祝福文本

## 6. 實施計劃

### 6.1 階段一：基礎版本（MVP）
- 功能：基本輸入、時間線視覺化（週視圖）、靜態壽命數據
- 時間：2-3 週

### 6.2 階段二：增強版本
- 功能：多時間單位視圖、LLM 壽命數據查詢、詩意祝福生成
- 時間：3-4 週

### 6.3 階段三：完整版本
- 功能：自訂里程碑、分享功能、多語言支持
- 時間：4-5 週

## 7. 技術挑戰與解決方案

### 7.1 挑戰：大規模時間單位渲染效能
**解決方案**：
- 虛擬滾動技術
- 僅渲染視窗內可見元素
- 使用 React.memo 優化渲染

### 7.2 挑戰：壽命數據準確性
**解決方案**：
- 結合多來源數據
- 定期更新靜態數據
- 使用 LLM 處理數據不一致

### 7.3 挑戰：生成內容質量
**解決方案**：
- 精心設計提示詞工程
- 定義清晰的輸出格式和風格指南
- 多樣性的參考範本

## 8. API 端點設計

### 8.1 主要端點
```
POST /api/calculate-life-timeline
{
  "birth_date": "1993-05-12",
  "gender": "male",
  "country": "Taiwan"
}

Response:
{
  "life_expectancy": 80.2,
  "time_units": {
    "days": {...},
    "weeks": {...},
    "months": {...},
    "years": {...}
  },
  "statistics": {
    "elapsed_percentage": 38.4,
    "current_life_stage": "職業生涯"
  },
  "poetic_blessing": "在時間的長河中，你已揮灑了生命的三分之一，如同台灣蘭陽平原上日出前的晨霧..."
}
```

## 9. 優化與擴展計劃

### 9.1 短期優化
- 響應式設計改進
- 動畫過渡效果
- 本地存儲功能

### 9.2 中期擴展
- 導出/分享功能
- 多語言支持
- 自訂生命階段

### 9.3 長期願景
- 用戶社區
- 年齡層和地區統計比較
- 生命規劃建議和資源

## 10. 總結

生命時間線應用旨在創建一個獨特的自我反思工具，透過數據視覺化和 AI 生成內容，幫助用戶理解生命的有限性和珍貴性。結合現代前端技術和 AI 能力，這個專案有潛力成為一個有意義且富有啟發性的個人工具，鼓勵用戶更加珍惜每一天並思考生命的意義和價值。
