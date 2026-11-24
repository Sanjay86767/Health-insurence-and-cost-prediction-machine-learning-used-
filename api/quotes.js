const connectToDatabase = require('../lib/mongodb');
const Quote = require('../models/Quote');
const jwt = require('jsonwebtoken');

const JWT_SECRET = process.env.JWT_SECRET || 'health_insurance_super_secret_jwt_key_2026';

function verifyToken(req) {
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return null;
  }
  const token = authHeader.split(' ')[1];
  try {
    return jwt.verify(token, JWT_SECRET);
  } catch (err) {
    return null;
  }
}

module.exports = async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  const user = verifyToken(req);
  if (!user) {
    return res.status(401).json({ success: false, message: 'Please sign in to view or save quotes' });
  }

  try {
    await connectToDatabase();

    if (req.method === 'POST') {
      const { age, gender, bmi, smoker, surgeries, conditions, quoteAmount, riskTier } = req.body || {};

      const quote = await Quote.create({
        userId: user.userId,
        userEmail: user.email,
        age,
        gender,
        bmi,
        smoker,
        surgeries,
        conditions: conditions || [],
        quoteAmount,
        riskTier
      });

      return res.status(201).json({ success: true, quote });
    }

    if (req.method === 'GET') {
      const quotes = await Quote.find({ userId: user.userId }).sort({ createdAt: -1 }).limit(20);
      return res.status(200).json({ success: true, quotes });
    }

    return res.status(405).json({ success: false, message: 'Method Not Allowed' });
  } catch (error) {
    console.error('Quotes API error:', error);
    return res.status(500).json({ success: false, message: error.message || 'Database error' });
  }
};
