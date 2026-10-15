const OKABE_ITO = {
  pronoun: [0xe6, 0x9f, 0x00],
  determiner: [0x56, 0xb4, 0xe9],
  preposition: [0x00, 0x9e, 0x73],
  conjunction: [0xf0, 0xe4, 0x42],
  auxiliary: [0x00, 0x72, 0xb2],
  adverb: [0xd5, 0x5e, 0x00],
  content: [0xcc, 0x79, 0xa7],
};

const MEMBERS = {
  pronoun: ["it", "i", "we", "he", "they", "she", "him", "my", "our", "it’s", "there"],
  determiner: ["the", "a", "an", "this", "all", "any"],
  preposition: ["to", "of", "in", "for", "at", "out", "as", "on"],
  conjunction: ["and", "but", "that"],
  auxiliary: ["is", "was", "be", "are", "will", "were", "am", "had", "has", "have", "do", "can"],
  adverb: ["not", "really", "very", "always", "so"],
  content: ["time", "think", "good", "new", "people"],
};

const CLASS_OF = new Map(Object.entries(MEMBERS).flatMap(([cls, words]) => words.map((w) => [w, cls])));

export const wordClass = (word) => CLASS_OF.get(word) ?? "other";

export const classColor = (cls, a = 1) => {
  const [r, g, b] = OKABE_ITO[cls] ?? [0x99, 0x99, 0x99];
  return `rgba(${r},${g},${b},${a})`;
};
