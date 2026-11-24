const mongoose = require('mongoose');

const QuoteSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  userEmail: {
    type: String,
    required: true
  },
  age: Number,
  gender: String,
  bmi: Number,
  smoker: Boolean,
  surgeries: Number,
  conditions: [String],
  quoteAmount: Number,
  riskTier: String,
  createdAt: {
    type: Date,
    default: Date.now
  }
});

module.exports = mongoose.models.Quote || mongoose.model('Quote', QuoteSchema);
