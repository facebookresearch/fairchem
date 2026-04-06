import {
  AppErrorBoundary,
  ContentReload,
  Document,
  HashLink,
  JUPYTER_RENDERERS,
  MyST,
  SEARCH_ATTRIBUTES_ORDERED,
  SearchFactoryProvider,
  SkipTo,
  extractField,
  getMetaTagsForSite,
  mergeRenderers,
  renderers,
  require_loaders,
  useGridSystemProvider
} from "/build/_shared/chunk-QYHIDYQN.js";
import {
  require_jsx_runtime
} from "/build/_shared/chunk-IW6XPN43.js";
import "/build/_shared/chunk-ZNGG4FXY.js";
import "/build/_shared/chunk-5XJWQ5WJ.js";
import "/build/_shared/chunk-SP2MKLPW.js";
import "/build/_shared/chunk-T7QTAV5N.js";
import "/build/_shared/chunk-2WW6JOYG.js";
import "/build/_shared/chunk-IMMBAB6Q.js";
import "/build/_shared/chunk-7SI5XUIQ.js";
import "/build/_shared/chunk-DOEQHMLC.js";
import {
  require_classnames
} from "/build/_shared/chunk-ZNG4DSCS.js";
import "/build/_shared/chunk-MWXHQG25.js";
import {
  matches,
  select,
  selectAll
} from "/build/_shared/chunk-Q6DHUCUI.js";
import {
  convert
} from "/build/_shared/chunk-LPSXN4QV.js";
import {
  Outlet,
  require_jsx_dev_runtime,
  useLoaderData
} from "/build/_shared/chunk-2RVQXRZB.js";
import {
  require_react
} from "/build/_shared/chunk-3RNZ6DIW.js";
import {
  __toESM
} from "/build/_shared/chunk-CGOEG7L2.js";

// app/styles/app.css
var app_default = "/build/_assets/app-4OILZK2U.css";

// ../../node_modules/thebe-core/dist/lib/thebe-core.css
var thebe_core_default = "/build/_assets/thebe-core-MA3QWTX6.css";

// app/root.tsx
var import_loaders = __toESM(require_loaders());

// ../../node_modules/minisearch/dist/es/index.js
var ENTRIES = "ENTRIES";
var KEYS = "KEYS";
var VALUES = "VALUES";
var LEAF = "";
var TreeIterator = class {
  constructor(set, type) {
    const node = set._tree;
    const keys = Array.from(node.keys());
    this.set = set;
    this._type = type;
    this._path = keys.length > 0 ? [{ node, keys }] : [];
  }
  next() {
    const value = this.dive();
    this.backtrack();
    return value;
  }
  dive() {
    if (this._path.length === 0) {
      return { done: true, value: void 0 };
    }
    const { node, keys } = last$1(this._path);
    if (last$1(keys) === LEAF) {
      return { done: false, value: this.result() };
    }
    const child = node.get(last$1(keys));
    this._path.push({ node: child, keys: Array.from(child.keys()) });
    return this.dive();
  }
  backtrack() {
    if (this._path.length === 0) {
      return;
    }
    const keys = last$1(this._path).keys;
    keys.pop();
    if (keys.length > 0) {
      return;
    }
    this._path.pop();
    this.backtrack();
  }
  key() {
    return this.set._prefix + this._path.map(({ keys }) => last$1(keys)).filter((key) => key !== LEAF).join("");
  }
  value() {
    return last$1(this._path).node.get(LEAF);
  }
  result() {
    switch (this._type) {
      case VALUES:
        return this.value();
      case KEYS:
        return this.key();
      default:
        return [this.key(), this.value()];
    }
  }
  [Symbol.iterator]() {
    return this;
  }
};
var last$1 = (array) => {
  return array[array.length - 1];
};
var fuzzySearch = (node, query, maxDistance) => {
  const results = /* @__PURE__ */ new Map();
  if (query === void 0)
    return results;
  const n = query.length + 1;
  const m = n + maxDistance;
  const matrix = new Uint8Array(m * n).fill(maxDistance + 1);
  for (let j = 0; j < n; ++j)
    matrix[j] = j;
  for (let i = 1; i < m; ++i)
    matrix[i * n] = i;
  recurse(node, query, maxDistance, results, matrix, 1, n, "");
  return results;
};
var recurse = (node, query, maxDistance, results, matrix, m, n, prefix) => {
  const offset = m * n;
  key:
    for (const key of node.keys()) {
      if (key === LEAF) {
        const distance = matrix[offset - 1];
        if (distance <= maxDistance) {
          results.set(prefix, [node.get(key), distance]);
        }
      } else {
        let i = m;
        for (let pos = 0; pos < key.length; ++pos, ++i) {
          const char = key[pos];
          const thisRowOffset = n * i;
          const prevRowOffset = thisRowOffset - n;
          let minDistance = matrix[thisRowOffset];
          const jmin = Math.max(0, i - maxDistance - 1);
          const jmax = Math.min(n - 1, i + maxDistance);
          for (let j = jmin; j < jmax; ++j) {
            const different = char !== query[j];
            const rpl = matrix[prevRowOffset + j] + +different;
            const del = matrix[prevRowOffset + j + 1] + 1;
            const ins = matrix[thisRowOffset + j] + 1;
            const dist = matrix[thisRowOffset + j + 1] = Math.min(rpl, del, ins);
            if (dist < minDistance)
              minDistance = dist;
          }
          if (minDistance > maxDistance) {
            continue key;
          }
        }
        recurse(node.get(key), query, maxDistance, results, matrix, i, n, prefix + key);
      }
    }
};
var SearchableMap = class {
  /**
   * The constructor is normally called without arguments, creating an empty
   * map. In order to create a {@link SearchableMap} from an iterable or from an
   * object, check {@link SearchableMap.from} and {@link
   * SearchableMap.fromObject}.
   *
   * The constructor arguments are for internal use, when creating derived
   * mutable views of a map at a prefix.
   */
  constructor(tree = /* @__PURE__ */ new Map(), prefix = "") {
    this._size = void 0;
    this._tree = tree;
    this._prefix = prefix;
  }
  /**
   * Creates and returns a mutable view of this {@link SearchableMap},
   * containing only entries that share the given prefix.
   *
   * ### Usage:
   *
   * ```javascript
   * let map = new SearchableMap()
   * map.set("unicorn", 1)
   * map.set("universe", 2)
   * map.set("university", 3)
   * map.set("unique", 4)
   * map.set("hello", 5)
   *
   * let uni = map.atPrefix("uni")
   * uni.get("unique") // => 4
   * uni.get("unicorn") // => 1
   * uni.get("hello") // => undefined
   *
   * let univer = map.atPrefix("univer")
   * univer.get("unique") // => undefined
   * univer.get("universe") // => 2
   * univer.get("university") // => 3
   * ```
   *
   * @param prefix  The prefix
   * @return A {@link SearchableMap} representing a mutable view of the original
   * Map at the given prefix
   */
  atPrefix(prefix) {
    if (!prefix.startsWith(this._prefix)) {
      throw new Error("Mismatched prefix");
    }
    const [node, path] = trackDown(this._tree, prefix.slice(this._prefix.length));
    if (node === void 0) {
      const [parentNode, key] = last(path);
      for (const k of parentNode.keys()) {
        if (k !== LEAF && k.startsWith(key)) {
          const node2 = /* @__PURE__ */ new Map();
          node2.set(k.slice(key.length), parentNode.get(k));
          return new SearchableMap(node2, prefix);
        }
      }
    }
    return new SearchableMap(node, prefix);
  }
  /**
   * @see https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Map/clear
   */
  clear() {
    this._size = void 0;
    this._tree.clear();
  }
  /**
   * @see https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Map/delete
   * @param key  Key to delete
   */
  delete(key) {
    this._size = void 0;
    return remove(this._tree, key);
  }
  /**
   * @see https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Map/entries
   * @return An iterator iterating through `[key, value]` entries.
   */
  entries() {
    return new TreeIterator(this, ENTRIES);
  }
  /**
   * @see https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Map/forEach
   * @param fn  Iteration function
   */
  forEach(fn) {
    for (const [key, value] of this) {
      fn(key, value, this);
    }
  }
  /**
   * Returns a Map of all the entries that have a key within the given edit
   * distance from the search key. The keys of the returned Map are the matching
   * keys, while the values are two-element arrays where the first element is
   * the value associated to the key, and the second is the edit distance of the
   * key to the search key.
   *
   * ### Usage:
   *
   * ```javascript
   * let map = new SearchableMap()
   * map.set('hello', 'world')
   * map.set('hell', 'yeah')
   * map.set('ciao', 'mondo')
   *
   * // Get all entries that match the key 'hallo' with a maximum edit distance of 2
   * map.fuzzyGet('hallo', 2)
   * // => Map(2) { 'hello' => ['world', 1], 'hell' => ['yeah', 2] }
   *
   * // In the example, the "hello" key has value "world" and edit distance of 1
   * // (change "e" to "a"), the key "hell" has value "yeah" and edit distance of 2
   * // (change "e" to "a", delete "o")
   * ```
   *
   * @param key  The search key
   * @param maxEditDistance  The maximum edit distance (Levenshtein)
   * @return A Map of the matching keys to their value and edit distance
   */
  fuzzyGet(key, maxEditDistance) {
    return fuzzySearch(this._tree, key, maxEditDistance);
  }
  /**
   * @see https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Map/get
   * @param key  Key to get
   * @return Value associated to the key, or `undefined` if the key is not
   * found.
   */
  get(key) {
    const node = lookup(this._tree, key);
    return node !== void 0 ? node.get(LEAF) : void 0;
  }
  /**
   * @see https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Map/has
   * @param key  Key
   * @return True if the key is in the map, false otherwise
   */
  has(key) {
    const node = lookup(this._tree, key);
    return node !== void 0 && node.has(LEAF);
  }
  /**
   * @see https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Map/keys
   * @return An `Iterable` iterating through keys
   */
  keys() {
    return new TreeIterator(this, KEYS);
  }
  /**
   * @see https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Map/set
   * @param key  Key to set
   * @param value  Value to associate to the key
   * @return The {@link SearchableMap} itself, to allow chaining
   */
  set(key, value) {
    if (typeof key !== "string") {
      throw new Error("key must be a string");
    }
    this._size = void 0;
    const node = createPath(this._tree, key);
    node.set(LEAF, value);
    return this;
  }
  /**
   * @see https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Map/size
   */
  get size() {
    if (this._size) {
      return this._size;
    }
    this._size = 0;
    const iter = this.entries();
    while (!iter.next().done)
      this._size += 1;
    return this._size;
  }
  /**
   * Updates the value at the given key using the provided function. The function
   * is called with the current value at the key, and its return value is used as
   * the new value to be set.
   *
   * ### Example:
   *
   * ```javascript
   * // Increment the current value by one
   * searchableMap.update('somekey', (currentValue) => currentValue == null ? 0 : currentValue + 1)
   * ```
   *
   * If the value at the given key is or will be an object, it might not require
   * re-assignment. In that case it is better to use `fetch()`, because it is
   * faster.
   *
   * @param key  The key to update
   * @param fn  The function used to compute the new value from the current one
   * @return The {@link SearchableMap} itself, to allow chaining
   */
  update(key, fn) {
    if (typeof key !== "string") {
      throw new Error("key must be a string");
    }
    this._size = void 0;
    const node = createPath(this._tree, key);
    node.set(LEAF, fn(node.get(LEAF)));
    return this;
  }
  /**
   * Fetches the value of the given key. If the value does not exist, calls the
   * given function to create a new value, which is inserted at the given key
   * and subsequently returned.
   *
   * ### Example:
   *
   * ```javascript
   * const map = searchableMap.fetch('somekey', () => new Map())
   * map.set('foo', 'bar')
   * ```
   *
   * @param key  The key to update
   * @param initial  A function that creates a new value if the key does not exist
   * @return The existing or new value at the given key
   */
  fetch(key, initial) {
    if (typeof key !== "string") {
      throw new Error("key must be a string");
    }
    this._size = void 0;
    const node = createPath(this._tree, key);
    let value = node.get(LEAF);
    if (value === void 0) {
      node.set(LEAF, value = initial());
    }
    return value;
  }
  /**
   * @see https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Map/values
   * @return An `Iterable` iterating through values.
   */
  values() {
    return new TreeIterator(this, VALUES);
  }
  /**
   * @see https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Map/@@iterator
   */
  [Symbol.iterator]() {
    return this.entries();
  }
  /**
   * Creates a {@link SearchableMap} from an `Iterable` of entries
   *
   * @param entries  Entries to be inserted in the {@link SearchableMap}
   * @return A new {@link SearchableMap} with the given entries
   */
  static from(entries) {
    const tree = new SearchableMap();
    for (const [key, value] of entries) {
      tree.set(key, value);
    }
    return tree;
  }
  /**
   * Creates a {@link SearchableMap} from the iterable properties of a JavaScript object
   *
   * @param object  Object of entries for the {@link SearchableMap}
   * @return A new {@link SearchableMap} with the given entries
   */
  static fromObject(object) {
    return SearchableMap.from(Object.entries(object));
  }
};
var trackDown = (tree, key, path = []) => {
  if (key.length === 0 || tree == null) {
    return [tree, path];
  }
  for (const k of tree.keys()) {
    if (k !== LEAF && key.startsWith(k)) {
      path.push([tree, k]);
      return trackDown(tree.get(k), key.slice(k.length), path);
    }
  }
  path.push([tree, key]);
  return trackDown(void 0, "", path);
};
var lookup = (tree, key) => {
  if (key.length === 0 || tree == null) {
    return tree;
  }
  for (const k of tree.keys()) {
    if (k !== LEAF && key.startsWith(k)) {
      return lookup(tree.get(k), key.slice(k.length));
    }
  }
};
var createPath = (node, key) => {
  const keyLength = key.length;
  outer:
    for (let pos = 0; node && pos < keyLength; ) {
      for (const k of node.keys()) {
        if (k !== LEAF && key[pos] === k[0]) {
          const len = Math.min(keyLength - pos, k.length);
          let offset = 1;
          while (offset < len && key[pos + offset] === k[offset])
            ++offset;
          const child2 = node.get(k);
          if (offset === k.length) {
            node = child2;
          } else {
            const intermediate = /* @__PURE__ */ new Map();
            intermediate.set(k.slice(offset), child2);
            node.set(key.slice(pos, pos + offset), intermediate);
            node.delete(k);
            node = intermediate;
          }
          pos += offset;
          continue outer;
        }
      }
      const child = /* @__PURE__ */ new Map();
      node.set(key.slice(pos), child);
      return child;
    }
  return node;
};
var remove = (tree, key) => {
  const [node, path] = trackDown(tree, key);
  if (node === void 0) {
    return;
  }
  node.delete(LEAF);
  if (node.size === 0) {
    cleanup(path);
  } else if (node.size === 1) {
    const [key2, value] = node.entries().next().value;
    merge(path, key2, value);
  }
};
var cleanup = (path) => {
  if (path.length === 0) {
    return;
  }
  const [node, key] = last(path);
  node.delete(key);
  if (node.size === 0) {
    cleanup(path.slice(0, -1));
  } else if (node.size === 1) {
    const [key2, value] = node.entries().next().value;
    if (key2 !== LEAF) {
      merge(path.slice(0, -1), key2, value);
    }
  }
};
var merge = (path, key, value) => {
  if (path.length === 0) {
    return;
  }
  const [node, nodeKey] = last(path);
  node.set(nodeKey + key, value);
  node.delete(nodeKey);
};
var last = (array) => {
  return array[array.length - 1];
};
var OR = "or";
var AND = "and";
var AND_NOT = "and_not";
var MiniSearch = class {
  /**
   * @param options  Configuration options
   *
   * ### Examples:
   *
   * ```javascript
   * // Create a search engine that indexes the 'title' and 'text' fields of your
   * // documents:
   * const miniSearch = new MiniSearch({ fields: ['title', 'text'] })
   * ```
   *
   * ### ID Field:
   *
   * ```javascript
   * // Your documents are assumed to include a unique 'id' field, but if you want
   * // to use a different field for document identification, you can set the
   * // 'idField' option:
   * const miniSearch = new MiniSearch({ idField: 'key', fields: ['title', 'text'] })
   * ```
   *
   * ### Options and defaults:
   *
   * ```javascript
   * // The full set of options (here with their default value) is:
   * const miniSearch = new MiniSearch({
   *   // idField: field that uniquely identifies a document
   *   idField: 'id',
   *
   *   // extractField: function used to get the value of a field in a document.
   *   // By default, it assumes the document is a flat object with field names as
   *   // property keys and field values as string property values, but custom logic
   *   // can be implemented by setting this option to a custom extractor function.
   *   extractField: (document, fieldName) => document[fieldName],
   *
   *   // tokenize: function used to split fields into individual terms. By
   *   // default, it is also used to tokenize search queries, unless a specific
   *   // `tokenize` search option is supplied. When tokenizing an indexed field,
   *   // the field name is passed as the second argument.
   *   tokenize: (string, _fieldName) => string.split(SPACE_OR_PUNCTUATION),
   *
   *   // processTerm: function used to process each tokenized term before
   *   // indexing. It can be used for stemming and normalization. Return a falsy
   *   // value in order to discard a term. By default, it is also used to process
   *   // search queries, unless a specific `processTerm` option is supplied as a
   *   // search option. When processing a term from a indexed field, the field
   *   // name is passed as the second argument.
   *   processTerm: (term, _fieldName) => term.toLowerCase(),
   *
   *   // searchOptions: default search options, see the `search` method for
   *   // details
   *   searchOptions: undefined,
   *
   *   // fields: document fields to be indexed. Mandatory, but not set by default
   *   fields: undefined
   *
   *   // storeFields: document fields to be stored and returned as part of the
   *   // search results.
   *   storeFields: []
   * })
   * ```
   */
  constructor(options) {
    if ((options === null || options === void 0 ? void 0 : options.fields) == null) {
      throw new Error('MiniSearch: option "fields" must be provided');
    }
    const autoVacuum = options.autoVacuum == null || options.autoVacuum === true ? defaultAutoVacuumOptions : options.autoVacuum;
    this._options = {
      ...defaultOptions,
      ...options,
      autoVacuum,
      searchOptions: { ...defaultSearchOptions, ...options.searchOptions || {} },
      autoSuggestOptions: { ...defaultAutoSuggestOptions, ...options.autoSuggestOptions || {} }
    };
    this._index = new SearchableMap();
    this._documentCount = 0;
    this._documentIds = /* @__PURE__ */ new Map();
    this._idToShortId = /* @__PURE__ */ new Map();
    this._fieldIds = {};
    this._fieldLength = /* @__PURE__ */ new Map();
    this._avgFieldLength = [];
    this._nextId = 0;
    this._storedFields = /* @__PURE__ */ new Map();
    this._dirtCount = 0;
    this._currentVacuum = null;
    this._enqueuedVacuum = null;
    this._enqueuedVacuumConditions = defaultVacuumConditions;
    this.addFields(this._options.fields);
  }
  /**
   * Adds a document to the index
   *
   * @param document  The document to be indexed
   */
  add(document2) {
    const { extractField: extractField2, stringifyField, tokenize, processTerm, fields, idField } = this._options;
    const id = extractField2(document2, idField);
    if (id == null) {
      throw new Error(`MiniSearch: document does not have ID field "${idField}"`);
    }
    if (this._idToShortId.has(id)) {
      throw new Error(`MiniSearch: duplicate ID ${id}`);
    }
    const shortDocumentId = this.addDocumentId(id);
    this.saveStoredFields(shortDocumentId, document2);
    for (const field of fields) {
      const fieldValue = extractField2(document2, field);
      if (fieldValue == null)
        continue;
      const tokens = tokenize(stringifyField(fieldValue, field), field);
      const fieldId = this._fieldIds[field];
      const uniqueTerms = new Set(tokens).size;
      this.addFieldLength(shortDocumentId, fieldId, this._documentCount - 1, uniqueTerms);
      for (const term of tokens) {
        const processedTerm = processTerm(term, field);
        if (Array.isArray(processedTerm)) {
          for (const t of processedTerm) {
            this.addTerm(fieldId, shortDocumentId, t);
          }
        } else if (processedTerm) {
          this.addTerm(fieldId, shortDocumentId, processedTerm);
        }
      }
    }
  }
  /**
   * Adds all the given documents to the index
   *
   * @param documents  An array of documents to be indexed
   */
  addAll(documents) {
    for (const document2 of documents)
      this.add(document2);
  }
  /**
   * Adds all the given documents to the index asynchronously.
   *
   * Returns a promise that resolves (to `undefined`) when the indexing is done.
   * This method is useful when index many documents, to avoid blocking the main
   * thread. The indexing is performed asynchronously and in chunks.
   *
   * @param documents  An array of documents to be indexed
   * @param options  Configuration options
   * @return A promise resolving to `undefined` when the indexing is done
   */
  addAllAsync(documents, options = {}) {
    const { chunkSize = 10 } = options;
    const acc = { chunk: [], promise: Promise.resolve() };
    const { chunk, promise } = documents.reduce(({ chunk: chunk2, promise: promise2 }, document2, i) => {
      chunk2.push(document2);
      if ((i + 1) % chunkSize === 0) {
        return {
          chunk: [],
          promise: promise2.then(() => new Promise((resolve) => setTimeout(resolve, 0))).then(() => this.addAll(chunk2))
        };
      } else {
        return { chunk: chunk2, promise: promise2 };
      }
    }, acc);
    return promise.then(() => this.addAll(chunk));
  }
  /**
   * Removes the given document from the index.
   *
   * The document to remove must NOT have changed between indexing and removal,
   * otherwise the index will be corrupted.
   *
   * This method requires passing the full document to be removed (not just the
   * ID), and immediately removes the document from the inverted index, allowing
   * memory to be released. A convenient alternative is {@link
   * MiniSearch#discard}, which needs only the document ID, and has the same
   * visible effect, but delays cleaning up the index until the next vacuuming.
   *
   * @param document  The document to be removed
   */
  remove(document2) {
    const { tokenize, processTerm, extractField: extractField2, stringifyField, fields, idField } = this._options;
    const id = extractField2(document2, idField);
    if (id == null) {
      throw new Error(`MiniSearch: document does not have ID field "${idField}"`);
    }
    const shortId = this._idToShortId.get(id);
    if (shortId == null) {
      throw new Error(`MiniSearch: cannot remove document with ID ${id}: it is not in the index`);
    }
    for (const field of fields) {
      const fieldValue = extractField2(document2, field);
      if (fieldValue == null)
        continue;
      const tokens = tokenize(stringifyField(fieldValue, field), field);
      const fieldId = this._fieldIds[field];
      const uniqueTerms = new Set(tokens).size;
      this.removeFieldLength(shortId, fieldId, this._documentCount, uniqueTerms);
      for (const term of tokens) {
        const processedTerm = processTerm(term, field);
        if (Array.isArray(processedTerm)) {
          for (const t of processedTerm) {
            this.removeTerm(fieldId, shortId, t);
          }
        } else if (processedTerm) {
          this.removeTerm(fieldId, shortId, processedTerm);
        }
      }
    }
    this._storedFields.delete(shortId);
    this._documentIds.delete(shortId);
    this._idToShortId.delete(id);
    this._fieldLength.delete(shortId);
    this._documentCount -= 1;
  }
  /**
   * Removes all the given documents from the index. If called with no arguments,
   * it removes _all_ documents from the index.
   *
   * @param documents  The documents to be removed. If this argument is omitted,
   * all documents are removed. Note that, for removing all documents, it is
   * more efficient to call this method with no arguments than to pass all
   * documents.
   */
  removeAll(documents) {
    if (documents) {
      for (const document2 of documents)
        this.remove(document2);
    } else if (arguments.length > 0) {
      throw new Error("Expected documents to be present. Omit the argument to remove all documents.");
    } else {
      this._index = new SearchableMap();
      this._documentCount = 0;
      this._documentIds = /* @__PURE__ */ new Map();
      this._idToShortId = /* @__PURE__ */ new Map();
      this._fieldLength = /* @__PURE__ */ new Map();
      this._avgFieldLength = [];
      this._storedFields = /* @__PURE__ */ new Map();
      this._nextId = 0;
    }
  }
  /**
   * Discards the document with the given ID, so it won't appear in search results
   *
   * It has the same visible effect of {@link MiniSearch.remove} (both cause the
   * document to stop appearing in searches), but a different effect on the
   * internal data structures:
   *
   *   - {@link MiniSearch#remove} requires passing the full document to be
   *   removed as argument, and removes it from the inverted index immediately.
   *
   *   - {@link MiniSearch#discard} instead only needs the document ID, and
   *   works by marking the current version of the document as discarded, so it
   *   is immediately ignored by searches. This is faster and more convenient
   *   than {@link MiniSearch#remove}, but the index is not immediately
   *   modified. To take care of that, vacuuming is performed after a certain
   *   number of documents are discarded, cleaning up the index and allowing
   *   memory to be released.
   *
   * After discarding a document, it is possible to re-add a new version, and
   * only the new version will appear in searches. In other words, discarding
   * and re-adding a document works exactly like removing and re-adding it. The
   * {@link MiniSearch.replace} method can also be used to replace a document
   * with a new version.
   *
   * #### Details about vacuuming
   *
   * Repetite calls to this method would leave obsolete document references in
   * the index, invisible to searches. Two mechanisms take care of cleaning up:
   * clean up during search, and vacuuming.
   *
   *   - Upon search, whenever a discarded ID is found (and ignored for the
   *   results), references to the discarded document are removed from the
   *   inverted index entries for the search terms. This ensures that subsequent
   *   searches for the same terms do not need to skip these obsolete references
   *   again.
   *
   *   - In addition, vacuuming is performed automatically by default (see the
   *   `autoVacuum` field in {@link Options}) after a certain number of
   *   documents are discarded. Vacuuming traverses all terms in the index,
   *   cleaning up all references to discarded documents. Vacuuming can also be
   *   triggered manually by calling {@link MiniSearch#vacuum}.
   *
   * @param id  The ID of the document to be discarded
   */
  discard(id) {
    const shortId = this._idToShortId.get(id);
    if (shortId == null) {
      throw new Error(`MiniSearch: cannot discard document with ID ${id}: it is not in the index`);
    }
    this._idToShortId.delete(id);
    this._documentIds.delete(shortId);
    this._storedFields.delete(shortId);
    (this._fieldLength.get(shortId) || []).forEach((fieldLength, fieldId) => {
      this.removeFieldLength(shortId, fieldId, this._documentCount, fieldLength);
    });
    this._fieldLength.delete(shortId);
    this._documentCount -= 1;
    this._dirtCount += 1;
    this.maybeAutoVacuum();
  }
  maybeAutoVacuum() {
    if (this._options.autoVacuum === false) {
      return;
    }
    const { minDirtFactor, minDirtCount, batchSize, batchWait } = this._options.autoVacuum;
    this.conditionalVacuum({ batchSize, batchWait }, { minDirtCount, minDirtFactor });
  }
  /**
   * Discards the documents with the given IDs, so they won't appear in search
   * results
   *
   * It is equivalent to calling {@link MiniSearch#discard} for all the given
   * IDs, but with the optimization of triggering at most one automatic
   * vacuuming at the end.
   *
   * Note: to remove all documents from the index, it is faster and more
   * convenient to call {@link MiniSearch.removeAll} with no argument, instead
   * of passing all IDs to this method.
   */
  discardAll(ids) {
    const autoVacuum = this._options.autoVacuum;
    try {
      this._options.autoVacuum = false;
      for (const id of ids) {
        this.discard(id);
      }
    } finally {
      this._options.autoVacuum = autoVacuum;
    }
    this.maybeAutoVacuum();
  }
  /**
   * It replaces an existing document with the given updated version
   *
   * It works by discarding the current version and adding the updated one, so
   * it is functionally equivalent to calling {@link MiniSearch#discard}
   * followed by {@link MiniSearch#add}. The ID of the updated document should
   * be the same as the original one.
   *
   * Since it uses {@link MiniSearch#discard} internally, this method relies on
   * vacuuming to clean up obsolete document references from the index, allowing
   * memory to be released (see {@link MiniSearch#discard}).
   *
   * @param updatedDocument  The updated document to replace the old version
   * with
   */
  replace(updatedDocument) {
    const { idField, extractField: extractField2 } = this._options;
    const id = extractField2(updatedDocument, idField);
    this.discard(id);
    this.add(updatedDocument);
  }
  /**
   * Triggers a manual vacuuming, cleaning up references to discarded documents
   * from the inverted index
   *
   * Vacuuming is only useful for applications that use the {@link
   * MiniSearch#discard} or {@link MiniSearch#replace} methods.
   *
   * By default, vacuuming is performed automatically when needed (controlled by
   * the `autoVacuum` field in {@link Options}), so there is usually no need to
   * call this method, unless one wants to make sure to perform vacuuming at a
   * specific moment.
   *
   * Vacuuming traverses all terms in the inverted index in batches, and cleans
   * up references to discarded documents from the posting list, allowing memory
   * to be released.
   *
   * The method takes an optional object as argument with the following keys:
   *
   *   - `batchSize`: the size of each batch (1000 by default)
   *
   *   - `batchWait`: the number of milliseconds to wait between batches (10 by
   *   default)
   *
   * On large indexes, vacuuming could have a non-negligible cost: batching
   * avoids blocking the thread for long, diluting this cost so that it is not
   * negatively affecting the application. Nonetheless, this method should only
   * be called when necessary, and relying on automatic vacuuming is usually
   * better.
   *
   * It returns a promise that resolves (to undefined) when the clean up is
   * completed. If vacuuming is already ongoing at the time this method is
   * called, a new one is enqueued immediately after the ongoing one, and a
   * corresponding promise is returned. However, no more than one vacuuming is
   * enqueued on top of the ongoing one, even if this method is called more
   * times (enqueuing multiple ones would be useless).
   *
   * @param options  Configuration options for the batch size and delay. See
   * {@link VacuumOptions}.
   */
  vacuum(options = {}) {
    return this.conditionalVacuum(options);
  }
  conditionalVacuum(options, conditions) {
    if (this._currentVacuum) {
      this._enqueuedVacuumConditions = this._enqueuedVacuumConditions && conditions;
      if (this._enqueuedVacuum != null) {
        return this._enqueuedVacuum;
      }
      this._enqueuedVacuum = this._currentVacuum.then(() => {
        const conditions2 = this._enqueuedVacuumConditions;
        this._enqueuedVacuumConditions = defaultVacuumConditions;
        return this.performVacuuming(options, conditions2);
      });
      return this._enqueuedVacuum;
    }
    if (this.vacuumConditionsMet(conditions) === false) {
      return Promise.resolve();
    }
    this._currentVacuum = this.performVacuuming(options);
    return this._currentVacuum;
  }
  async performVacuuming(options, conditions) {
    const initialDirtCount = this._dirtCount;
    if (this.vacuumConditionsMet(conditions)) {
      const batchSize = options.batchSize || defaultVacuumOptions.batchSize;
      const batchWait = options.batchWait || defaultVacuumOptions.batchWait;
      let i = 1;
      for (const [term, fieldsData] of this._index) {
        for (const [fieldId, fieldIndex] of fieldsData) {
          for (const [shortId] of fieldIndex) {
            if (this._documentIds.has(shortId)) {
              continue;
            }
            if (fieldIndex.size <= 1) {
              fieldsData.delete(fieldId);
            } else {
              fieldIndex.delete(shortId);
            }
          }
        }
        if (this._index.get(term).size === 0) {
          this._index.delete(term);
        }
        if (i % batchSize === 0) {
          await new Promise((resolve) => setTimeout(resolve, batchWait));
        }
        i += 1;
      }
      this._dirtCount -= initialDirtCount;
    }
    await null;
    this._currentVacuum = this._enqueuedVacuum;
    this._enqueuedVacuum = null;
  }
  vacuumConditionsMet(conditions) {
    if (conditions == null) {
      return true;
    }
    let { minDirtCount, minDirtFactor } = conditions;
    minDirtCount = minDirtCount || defaultAutoVacuumOptions.minDirtCount;
    minDirtFactor = minDirtFactor || defaultAutoVacuumOptions.minDirtFactor;
    return this.dirtCount >= minDirtCount && this.dirtFactor >= minDirtFactor;
  }
  /**
   * Is `true` if a vacuuming operation is ongoing, `false` otherwise
   */
  get isVacuuming() {
    return this._currentVacuum != null;
  }
  /**
   * The number of documents discarded since the most recent vacuuming
   */
  get dirtCount() {
    return this._dirtCount;
  }
  /**
   * A number between 0 and 1 giving an indication about the proportion of
   * documents that are discarded, and can therefore be cleaned up by vacuuming.
   * A value close to 0 means that the index is relatively clean, while a higher
   * value means that the index is relatively dirty, and vacuuming could release
   * memory.
   */
  get dirtFactor() {
    return this._dirtCount / (1 + this._documentCount + this._dirtCount);
  }
  /**
   * Returns `true` if a document with the given ID is present in the index and
   * available for search, `false` otherwise
   *
   * @param id  The document ID
   */
  has(id) {
    return this._idToShortId.has(id);
  }
  /**
   * Returns the stored fields (as configured in the `storeFields` constructor
   * option) for the given document ID. Returns `undefined` if the document is
   * not present in the index.
   *
   * @param id  The document ID
   */
  getStoredFields(id) {
    const shortId = this._idToShortId.get(id);
    if (shortId == null) {
      return void 0;
    }
    return this._storedFields.get(shortId);
  }
  /**
   * Search for documents matching the given search query.
   *
   * The result is a list of scored document IDs matching the query, sorted by
   * descending score, and each including data about which terms were matched and
   * in which fields.
   *
   * ### Basic usage:
   *
   * ```javascript
   * // Search for "zen art motorcycle" with default options: terms have to match
   * // exactly, and individual terms are joined with OR
   * miniSearch.search('zen art motorcycle')
   * // => [ { id: 2, score: 2.77258, match: { ... } }, { id: 4, score: 1.38629, match: { ... } } ]
   * ```
   *
   * ### Restrict search to specific fields:
   *
   * ```javascript
   * // Search only in the 'title' field
   * miniSearch.search('zen', { fields: ['title'] })
   * ```
   *
   * ### Field boosting:
   *
   * ```javascript
   * // Boost a field
   * miniSearch.search('zen', { boost: { title: 2 } })
   * ```
   *
   * ### Prefix search:
   *
   * ```javascript
   * // Search for "moto" with prefix search (it will match documents
   * // containing terms that start with "moto" or "neuro")
   * miniSearch.search('moto neuro', { prefix: true })
   * ```
   *
   * ### Fuzzy search:
   *
   * ```javascript
   * // Search for "ismael" with fuzzy search (it will match documents containing
   * // terms similar to "ismael", with a maximum edit distance of 0.2 term.length
   * // (rounded to nearest integer)
   * miniSearch.search('ismael', { fuzzy: 0.2 })
   * ```
   *
   * ### Combining strategies:
   *
   * ```javascript
   * // Mix of exact match, prefix search, and fuzzy search
   * miniSearch.search('ismael mob', {
   *  prefix: true,
   *  fuzzy: 0.2
   * })
   * ```
   *
   * ### Advanced prefix and fuzzy search:
   *
   * ```javascript
   * // Perform fuzzy and prefix search depending on the search term. Here
   * // performing prefix and fuzzy search only on terms longer than 3 characters
   * miniSearch.search('ismael mob', {
   *  prefix: term => term.length > 3
   *  fuzzy: term => term.length > 3 ? 0.2 : null
   * })
   * ```
   *
   * ### Combine with AND:
   *
   * ```javascript
   * // Combine search terms with AND (to match only documents that contain both
   * // "motorcycle" and "art")
   * miniSearch.search('motorcycle art', { combineWith: 'AND' })
   * ```
   *
   * ### Combine with AND_NOT:
   *
   * There is also an AND_NOT combinator, that finds documents that match the
   * first term, but do not match any of the other terms. This combinator is
   * rarely useful with simple queries, and is meant to be used with advanced
   * query combinations (see later for more details).
   *
   * ### Filtering results:
   *
   * ```javascript
   * // Filter only results in the 'fiction' category (assuming that 'category'
   * // is a stored field)
   * miniSearch.search('motorcycle art', {
   *   filter: (result) => result.category === 'fiction'
   * })
   * ```
   *
   * ### Wildcard query
   *
   * Searching for an empty string (assuming the default tokenizer) returns no
   * results. Sometimes though, one needs to match all documents, like in a
   * "wildcard" search. This is possible by passing the special value
   * {@link MiniSearch.wildcard} as the query:
   *
   * ```javascript
   * // Return search results for all documents
   * miniSearch.search(MiniSearch.wildcard)
   * ```
   *
   * Note that search options such as `filter` and `boostDocument` are still
   * applied, influencing which results are returned, and their order:
   *
   * ```javascript
   * // Return search results for all documents in the 'fiction' category
   * miniSearch.search(MiniSearch.wildcard, {
   *   filter: (result) => result.category === 'fiction'
   * })
   * ```
   *
   * ### Advanced combination of queries:
   *
   * It is possible to combine different subqueries with OR, AND, and AND_NOT,
   * and even with different search options, by passing a query expression
   * tree object as the first argument, instead of a string.
   *
   * ```javascript
   * // Search for documents that contain "zen" and ("motorcycle" or "archery")
   * miniSearch.search({
   *   combineWith: 'AND',
   *   queries: [
   *     'zen',
   *     {
   *       combineWith: 'OR',
   *       queries: ['motorcycle', 'archery']
   *     }
   *   ]
   * })
   *
   * // Search for documents that contain ("apple" or "pear") but not "juice" and
   * // not "tree"
   * miniSearch.search({
   *   combineWith: 'AND_NOT',
   *   queries: [
   *     {
   *       combineWith: 'OR',
   *       queries: ['apple', 'pear']
   *     },
   *     'juice',
   *     'tree'
   *   ]
   * })
   * ```
   *
   * Each node in the expression tree can be either a string, or an object that
   * supports all {@link SearchOptions} fields, plus a `queries` array field for
   * subqueries.
   *
   * Note that, while this can become complicated to do by hand for complex or
   * deeply nested queries, it provides a formalized expression tree API for
   * external libraries that implement a parser for custom query languages.
   *
   * @param query  Search query
   * @param searchOptions  Search options. Each option, if not given, defaults to the corresponding value of `searchOptions` given to the constructor, or to the library default.
   */
  search(query, searchOptions = {}) {
    const { searchOptions: globalSearchOptions } = this._options;
    const searchOptionsWithDefaults = { ...globalSearchOptions, ...searchOptions };
    const rawResults = this.executeQuery(query, searchOptions);
    const results = [];
    for (const [docId, { score, terms, match }] of rawResults) {
      const quality = terms.length || 1;
      const result = {
        id: this._documentIds.get(docId),
        score: score * quality,
        terms: Object.keys(match),
        queryTerms: terms,
        match
      };
      Object.assign(result, this._storedFields.get(docId));
      if (searchOptionsWithDefaults.filter == null || searchOptionsWithDefaults.filter(result)) {
        results.push(result);
      }
    }
    if (query === MiniSearch.wildcard && searchOptionsWithDefaults.boostDocument == null) {
      return results;
    }
    results.sort(byScore);
    return results;
  }
  /**
   * Provide suggestions for the given search query
   *
   * The result is a list of suggested modified search queries, derived from the
   * given search query, each with a relevance score, sorted by descending score.
   *
   * By default, it uses the same options used for search, except that by
   * default it performs prefix search on the last term of the query, and
   * combine terms with `'AND'` (requiring all query terms to match). Custom
   * options can be passed as a second argument. Defaults can be changed upon
   * calling the {@link MiniSearch} constructor, by passing a
   * `autoSuggestOptions` option.
   *
   * ### Basic usage:
   *
   * ```javascript
   * // Get suggestions for 'neuro':
   * miniSearch.autoSuggest('neuro')
   * // => [ { suggestion: 'neuromancer', terms: [ 'neuromancer' ], score: 0.46240 } ]
   * ```
   *
   * ### Multiple words:
   *
   * ```javascript
   * // Get suggestions for 'zen ar':
   * miniSearch.autoSuggest('zen ar')
   * // => [
   * //  { suggestion: 'zen archery art', terms: [ 'zen', 'archery', 'art' ], score: 1.73332 },
   * //  { suggestion: 'zen art', terms: [ 'zen', 'art' ], score: 1.21313 }
   * // ]
   * ```
   *
   * ### Fuzzy suggestions:
   *
   * ```javascript
   * // Correct spelling mistakes using fuzzy search:
   * miniSearch.autoSuggest('neromancer', { fuzzy: 0.2 })
   * // => [ { suggestion: 'neuromancer', terms: [ 'neuromancer' ], score: 1.03998 } ]
   * ```
   *
   * ### Filtering:
   *
   * ```javascript
   * // Get suggestions for 'zen ar', but only within the 'fiction' category
   * // (assuming that 'category' is a stored field):
   * miniSearch.autoSuggest('zen ar', {
   *   filter: (result) => result.category === 'fiction'
   * })
   * // => [
   * //  { suggestion: 'zen archery art', terms: [ 'zen', 'archery', 'art' ], score: 1.73332 },
   * //  { suggestion: 'zen art', terms: [ 'zen', 'art' ], score: 1.21313 }
   * // ]
   * ```
   *
   * @param queryString  Query string to be expanded into suggestions
   * @param options  Search options. The supported options and default values
   * are the same as for the {@link MiniSearch#search} method, except that by
   * default prefix search is performed on the last term in the query, and terms
   * are combined with `'AND'`.
   * @return  A sorted array of suggestions sorted by relevance score.
   */
  autoSuggest(queryString, options = {}) {
    options = { ...this._options.autoSuggestOptions, ...options };
    const suggestions = /* @__PURE__ */ new Map();
    for (const { score, terms } of this.search(queryString, options)) {
      const phrase = terms.join(" ");
      const suggestion = suggestions.get(phrase);
      if (suggestion != null) {
        suggestion.score += score;
        suggestion.count += 1;
      } else {
        suggestions.set(phrase, { score, terms, count: 1 });
      }
    }
    const results = [];
    for (const [suggestion, { score, terms, count }] of suggestions) {
      results.push({ suggestion, terms, score: score / count });
    }
    results.sort(byScore);
    return results;
  }
  /**
   * Total number of documents available to search
   */
  get documentCount() {
    return this._documentCount;
  }
  /**
   * Number of terms in the index
   */
  get termCount() {
    return this._index.size;
  }
  /**
   * Deserializes a JSON index (serialized with `JSON.stringify(miniSearch)`)
   * and instantiates a MiniSearch instance. It should be given the same options
   * originally used when serializing the index.
   *
   * ### Usage:
   *
   * ```javascript
   * // If the index was serialized with:
   * let miniSearch = new MiniSearch({ fields: ['title', 'text'] })
   * miniSearch.addAll(documents)
   *
   * const json = JSON.stringify(miniSearch)
   * // It can later be deserialized like this:
   * miniSearch = MiniSearch.loadJSON(json, { fields: ['title', 'text'] })
   * ```
   *
   * @param json  JSON-serialized index
   * @param options  configuration options, same as the constructor
   * @return An instance of MiniSearch deserialized from the given JSON.
   */
  static loadJSON(json, options) {
    if (options == null) {
      throw new Error("MiniSearch: loadJSON should be given the same options used when serializing the index");
    }
    return this.loadJS(JSON.parse(json), options);
  }
  /**
   * Async equivalent of {@link MiniSearch.loadJSON}
   *
   * This function is an alternative to {@link MiniSearch.loadJSON} that returns
   * a promise, and loads the index in batches, leaving pauses between them to avoid
   * blocking the main thread. It tends to be slower than the synchronous
   * version, but does not block the main thread, so it can be a better choice
   * when deserializing very large indexes.
   *
   * @param json  JSON-serialized index
   * @param options  configuration options, same as the constructor
   * @return A Promise that will resolve to an instance of MiniSearch deserialized from the given JSON.
   */
  static async loadJSONAsync(json, options) {
    if (options == null) {
      throw new Error("MiniSearch: loadJSON should be given the same options used when serializing the index");
    }
    return this.loadJSAsync(JSON.parse(json), options);
  }
  /**
   * Returns the default value of an option. It will throw an error if no option
   * with the given name exists.
   *
   * @param optionName  Name of the option
   * @return The default value of the given option
   *
   * ### Usage:
   *
   * ```javascript
   * // Get default tokenizer
   * MiniSearch.getDefault('tokenize')
   *
   * // Get default term processor
   * MiniSearch.getDefault('processTerm')
   *
   * // Unknown options will throw an error
   * MiniSearch.getDefault('notExisting')
   * // => throws 'MiniSearch: unknown option "notExisting"'
   * ```
   */
  static getDefault(optionName) {
    if (defaultOptions.hasOwnProperty(optionName)) {
      return getOwnProperty(defaultOptions, optionName);
    } else {
      throw new Error(`MiniSearch: unknown option "${optionName}"`);
    }
  }
  /**
   * @ignore
   */
  static loadJS(js, options) {
    const { index, documentIds, fieldLength, storedFields, serializationVersion } = js;
    const miniSearch = this.instantiateMiniSearch(js, options);
    miniSearch._documentIds = objectToNumericMap(documentIds);
    miniSearch._fieldLength = objectToNumericMap(fieldLength);
    miniSearch._storedFields = objectToNumericMap(storedFields);
    for (const [shortId, id] of miniSearch._documentIds) {
      miniSearch._idToShortId.set(id, shortId);
    }
    for (const [term, data] of index) {
      const dataMap = /* @__PURE__ */ new Map();
      for (const fieldId of Object.keys(data)) {
        let indexEntry = data[fieldId];
        if (serializationVersion === 1) {
          indexEntry = indexEntry.ds;
        }
        dataMap.set(parseInt(fieldId, 10), objectToNumericMap(indexEntry));
      }
      miniSearch._index.set(term, dataMap);
    }
    return miniSearch;
  }
  /**
   * @ignore
   */
  static async loadJSAsync(js, options) {
    const { index, documentIds, fieldLength, storedFields, serializationVersion } = js;
    const miniSearch = this.instantiateMiniSearch(js, options);
    miniSearch._documentIds = await objectToNumericMapAsync(documentIds);
    miniSearch._fieldLength = await objectToNumericMapAsync(fieldLength);
    miniSearch._storedFields = await objectToNumericMapAsync(storedFields);
    for (const [shortId, id] of miniSearch._documentIds) {
      miniSearch._idToShortId.set(id, shortId);
    }
    let count = 0;
    for (const [term, data] of index) {
      const dataMap = /* @__PURE__ */ new Map();
      for (const fieldId of Object.keys(data)) {
        let indexEntry = data[fieldId];
        if (serializationVersion === 1) {
          indexEntry = indexEntry.ds;
        }
        dataMap.set(parseInt(fieldId, 10), await objectToNumericMapAsync(indexEntry));
      }
      if (++count % 1e3 === 0)
        await wait(0);
      miniSearch._index.set(term, dataMap);
    }
    return miniSearch;
  }
  /**
   * @ignore
   */
  static instantiateMiniSearch(js, options) {
    const { documentCount, nextId, fieldIds, averageFieldLength, dirtCount, serializationVersion } = js;
    if (serializationVersion !== 1 && serializationVersion !== 2) {
      throw new Error("MiniSearch: cannot deserialize an index created with an incompatible version");
    }
    const miniSearch = new MiniSearch(options);
    miniSearch._documentCount = documentCount;
    miniSearch._nextId = nextId;
    miniSearch._idToShortId = /* @__PURE__ */ new Map();
    miniSearch._fieldIds = fieldIds;
    miniSearch._avgFieldLength = averageFieldLength;
    miniSearch._dirtCount = dirtCount || 0;
    miniSearch._index = new SearchableMap();
    return miniSearch;
  }
  /**
   * @ignore
   */
  executeQuery(query, searchOptions = {}) {
    if (query === MiniSearch.wildcard) {
      return this.executeWildcardQuery(searchOptions);
    }
    if (typeof query !== "string") {
      const options2 = { ...searchOptions, ...query, queries: void 0 };
      const results2 = query.queries.map((subquery) => this.executeQuery(subquery, options2));
      return this.combineResults(results2, options2.combineWith);
    }
    const { tokenize, processTerm, searchOptions: globalSearchOptions } = this._options;
    const options = { tokenize, processTerm, ...globalSearchOptions, ...searchOptions };
    const { tokenize: searchTokenize, processTerm: searchProcessTerm } = options;
    const terms = searchTokenize(query).flatMap((term) => searchProcessTerm(term)).filter((term) => !!term);
    const queries = terms.map(termToQuerySpec(options));
    const results = queries.map((query2) => this.executeQuerySpec(query2, options));
    return this.combineResults(results, options.combineWith);
  }
  /**
   * @ignore
   */
  executeQuerySpec(query, searchOptions) {
    const options = { ...this._options.searchOptions, ...searchOptions };
    const boosts = (options.fields || this._options.fields).reduce((boosts2, field) => ({ ...boosts2, [field]: getOwnProperty(options.boost, field) || 1 }), {});
    const { boostDocument, weights, maxFuzzy, bm25: bm25params } = options;
    const { fuzzy: fuzzyWeight, prefix: prefixWeight } = { ...defaultSearchOptions.weights, ...weights };
    const data = this._index.get(query.term);
    const results = this.termResults(query.term, query.term, 1, query.termBoost, data, boosts, boostDocument, bm25params);
    let prefixMatches;
    let fuzzyMatches;
    if (query.prefix) {
      prefixMatches = this._index.atPrefix(query.term);
    }
    if (query.fuzzy) {
      const fuzzy = query.fuzzy === true ? 0.2 : query.fuzzy;
      const maxDistance = fuzzy < 1 ? Math.min(maxFuzzy, Math.round(query.term.length * fuzzy)) : fuzzy;
      if (maxDistance)
        fuzzyMatches = this._index.fuzzyGet(query.term, maxDistance);
    }
    if (prefixMatches) {
      for (const [term, data2] of prefixMatches) {
        const distance = term.length - query.term.length;
        if (!distance) {
          continue;
        }
        fuzzyMatches === null || fuzzyMatches === void 0 ? void 0 : fuzzyMatches.delete(term);
        const weight = prefixWeight * term.length / (term.length + 0.3 * distance);
        this.termResults(query.term, term, weight, query.termBoost, data2, boosts, boostDocument, bm25params, results);
      }
    }
    if (fuzzyMatches) {
      for (const term of fuzzyMatches.keys()) {
        const [data2, distance] = fuzzyMatches.get(term);
        if (!distance) {
          continue;
        }
        const weight = fuzzyWeight * term.length / (term.length + distance);
        this.termResults(query.term, term, weight, query.termBoost, data2, boosts, boostDocument, bm25params, results);
      }
    }
    return results;
  }
  /**
   * @ignore
   */
  executeWildcardQuery(searchOptions) {
    const results = /* @__PURE__ */ new Map();
    const options = { ...this._options.searchOptions, ...searchOptions };
    for (const [shortId, id] of this._documentIds) {
      const score = options.boostDocument ? options.boostDocument(id, "", this._storedFields.get(shortId)) : 1;
      results.set(shortId, {
        score,
        terms: [],
        match: {}
      });
    }
    return results;
  }
  /**
   * @ignore
   */
  combineResults(results, combineWith = OR) {
    if (results.length === 0) {
      return /* @__PURE__ */ new Map();
    }
    const operator = combineWith.toLowerCase();
    const combinator = combinators[operator];
    if (!combinator) {
      throw new Error(`Invalid combination operator: ${combineWith}`);
    }
    return results.reduce(combinator) || /* @__PURE__ */ new Map();
  }
  /**
   * Allows serialization of the index to JSON, to possibly store it and later
   * deserialize it with {@link MiniSearch.loadJSON}.
   *
   * Normally one does not directly call this method, but rather call the
   * standard JavaScript `JSON.stringify()` passing the {@link MiniSearch}
   * instance, and JavaScript will internally call this method. Upon
   * deserialization, one must pass to {@link MiniSearch.loadJSON} the same
   * options used to create the original instance that was serialized.
   *
   * ### Usage:
   *
   * ```javascript
   * // Serialize the index:
   * let miniSearch = new MiniSearch({ fields: ['title', 'text'] })
   * miniSearch.addAll(documents)
   * const json = JSON.stringify(miniSearch)
   *
   * // Later, to deserialize it:
   * miniSearch = MiniSearch.loadJSON(json, { fields: ['title', 'text'] })
   * ```
   *
   * @return A plain-object serializable representation of the search index.
   */
  toJSON() {
    const index = [];
    for (const [term, fieldIndex] of this._index) {
      const data = {};
      for (const [fieldId, freqs] of fieldIndex) {
        data[fieldId] = Object.fromEntries(freqs);
      }
      index.push([term, data]);
    }
    return {
      documentCount: this._documentCount,
      nextId: this._nextId,
      documentIds: Object.fromEntries(this._documentIds),
      fieldIds: this._fieldIds,
      fieldLength: Object.fromEntries(this._fieldLength),
      averageFieldLength: this._avgFieldLength,
      storedFields: Object.fromEntries(this._storedFields),
      dirtCount: this._dirtCount,
      index,
      serializationVersion: 2
    };
  }
  /**
   * @ignore
   */
  termResults(sourceTerm, derivedTerm, termWeight, termBoost, fieldTermData, fieldBoosts, boostDocumentFn, bm25params, results = /* @__PURE__ */ new Map()) {
    if (fieldTermData == null)
      return results;
    for (const field of Object.keys(fieldBoosts)) {
      const fieldBoost = fieldBoosts[field];
      const fieldId = this._fieldIds[field];
      const fieldTermFreqs = fieldTermData.get(fieldId);
      if (fieldTermFreqs == null)
        continue;
      let matchingFields = fieldTermFreqs.size;
      const avgFieldLength = this._avgFieldLength[fieldId];
      for (const docId of fieldTermFreqs.keys()) {
        if (!this._documentIds.has(docId)) {
          this.removeTerm(fieldId, docId, derivedTerm);
          matchingFields -= 1;
          continue;
        }
        const docBoost = boostDocumentFn ? boostDocumentFn(this._documentIds.get(docId), derivedTerm, this._storedFields.get(docId)) : 1;
        if (!docBoost)
          continue;
        const termFreq = fieldTermFreqs.get(docId);
        const fieldLength = this._fieldLength.get(docId)[fieldId];
        const rawScore = calcBM25Score(termFreq, matchingFields, this._documentCount, fieldLength, avgFieldLength, bm25params);
        const weightedScore = termWeight * termBoost * fieldBoost * docBoost * rawScore;
        const result = results.get(docId);
        if (result) {
          result.score += weightedScore;
          assignUniqueTerm(result.terms, sourceTerm);
          const match = getOwnProperty(result.match, derivedTerm);
          if (match) {
            match.push(field);
          } else {
            result.match[derivedTerm] = [field];
          }
        } else {
          results.set(docId, {
            score: weightedScore,
            terms: [sourceTerm],
            match: { [derivedTerm]: [field] }
          });
        }
      }
    }
    return results;
  }
  /**
   * @ignore
   */
  addTerm(fieldId, documentId, term) {
    const indexData = this._index.fetch(term, createMap);
    let fieldIndex = indexData.get(fieldId);
    if (fieldIndex == null) {
      fieldIndex = /* @__PURE__ */ new Map();
      fieldIndex.set(documentId, 1);
      indexData.set(fieldId, fieldIndex);
    } else {
      const docs = fieldIndex.get(documentId);
      fieldIndex.set(documentId, (docs || 0) + 1);
    }
  }
  /**
   * @ignore
   */
  removeTerm(fieldId, documentId, term) {
    if (!this._index.has(term)) {
      this.warnDocumentChanged(documentId, fieldId, term);
      return;
    }
    const indexData = this._index.fetch(term, createMap);
    const fieldIndex = indexData.get(fieldId);
    if (fieldIndex == null || fieldIndex.get(documentId) == null) {
      this.warnDocumentChanged(documentId, fieldId, term);
    } else if (fieldIndex.get(documentId) <= 1) {
      if (fieldIndex.size <= 1) {
        indexData.delete(fieldId);
      } else {
        fieldIndex.delete(documentId);
      }
    } else {
      fieldIndex.set(documentId, fieldIndex.get(documentId) - 1);
    }
    if (this._index.get(term).size === 0) {
      this._index.delete(term);
    }
  }
  /**
   * @ignore
   */
  warnDocumentChanged(shortDocumentId, fieldId, term) {
    for (const fieldName of Object.keys(this._fieldIds)) {
      if (this._fieldIds[fieldName] === fieldId) {
        this._options.logger("warn", `MiniSearch: document with ID ${this._documentIds.get(shortDocumentId)} has changed before removal: term "${term}" was not present in field "${fieldName}". Removing a document after it has changed can corrupt the index!`, "version_conflict");
        return;
      }
    }
  }
  /**
   * @ignore
   */
  addDocumentId(documentId) {
    const shortDocumentId = this._nextId;
    this._idToShortId.set(documentId, shortDocumentId);
    this._documentIds.set(shortDocumentId, documentId);
    this._documentCount += 1;
    this._nextId += 1;
    return shortDocumentId;
  }
  /**
   * @ignore
   */
  addFields(fields) {
    for (let i = 0; i < fields.length; i++) {
      this._fieldIds[fields[i]] = i;
    }
  }
  /**
   * @ignore
   */
  addFieldLength(documentId, fieldId, count, length) {
    let fieldLengths = this._fieldLength.get(documentId);
    if (fieldLengths == null)
      this._fieldLength.set(documentId, fieldLengths = []);
    fieldLengths[fieldId] = length;
    const averageFieldLength = this._avgFieldLength[fieldId] || 0;
    const totalFieldLength = averageFieldLength * count + length;
    this._avgFieldLength[fieldId] = totalFieldLength / (count + 1);
  }
  /**
   * @ignore
   */
  removeFieldLength(documentId, fieldId, count, length) {
    if (count === 1) {
      this._avgFieldLength[fieldId] = 0;
      return;
    }
    const totalFieldLength = this._avgFieldLength[fieldId] * count - length;
    this._avgFieldLength[fieldId] = totalFieldLength / (count - 1);
  }
  /**
   * @ignore
   */
  saveStoredFields(documentId, doc) {
    const { storeFields, extractField: extractField2 } = this._options;
    if (storeFields == null || storeFields.length === 0) {
      return;
    }
    let documentFields = this._storedFields.get(documentId);
    if (documentFields == null)
      this._storedFields.set(documentId, documentFields = {});
    for (const fieldName of storeFields) {
      const fieldValue = extractField2(doc, fieldName);
      if (fieldValue !== void 0)
        documentFields[fieldName] = fieldValue;
    }
  }
};
MiniSearch.wildcard = Symbol("*");
var getOwnProperty = (object, property) => Object.prototype.hasOwnProperty.call(object, property) ? object[property] : void 0;
var combinators = {
  [OR]: (a, b) => {
    for (const docId of b.keys()) {
      const existing = a.get(docId);
      if (existing == null) {
        a.set(docId, b.get(docId));
      } else {
        const { score, terms, match } = b.get(docId);
        existing.score = existing.score + score;
        existing.match = Object.assign(existing.match, match);
        assignUniqueTerms(existing.terms, terms);
      }
    }
    return a;
  },
  [AND]: (a, b) => {
    const combined = /* @__PURE__ */ new Map();
    for (const docId of b.keys()) {
      const existing = a.get(docId);
      if (existing == null)
        continue;
      const { score, terms, match } = b.get(docId);
      assignUniqueTerms(existing.terms, terms);
      combined.set(docId, {
        score: existing.score + score,
        terms: existing.terms,
        match: Object.assign(existing.match, match)
      });
    }
    return combined;
  },
  [AND_NOT]: (a, b) => {
    for (const docId of b.keys())
      a.delete(docId);
    return a;
  }
};
var defaultBM25params = { k: 1.2, b: 0.7, d: 0.5 };
var calcBM25Score = (termFreq, matchingCount, totalCount, fieldLength, avgFieldLength, bm25params) => {
  const { k, b, d } = bm25params;
  const invDocFreq = Math.log(1 + (totalCount - matchingCount + 0.5) / (matchingCount + 0.5));
  return invDocFreq * (d + termFreq * (k + 1) / (termFreq + k * (1 - b + b * fieldLength / avgFieldLength)));
};
var termToQuerySpec = (options) => (term, i, terms) => {
  const fuzzy = typeof options.fuzzy === "function" ? options.fuzzy(term, i, terms) : options.fuzzy || false;
  const prefix = typeof options.prefix === "function" ? options.prefix(term, i, terms) : options.prefix === true;
  const termBoost = typeof options.boostTerm === "function" ? options.boostTerm(term, i, terms) : 1;
  return { term, fuzzy, prefix, termBoost };
};
var defaultOptions = {
  idField: "id",
  extractField: (document2, fieldName) => document2[fieldName],
  stringifyField: (fieldValue, fieldName) => fieldValue.toString(),
  tokenize: (text) => text.split(SPACE_OR_PUNCTUATION),
  processTerm: (term) => term.toLowerCase(),
  fields: void 0,
  searchOptions: void 0,
  storeFields: [],
  logger: (level, message) => {
    if (typeof (console === null || console === void 0 ? void 0 : console[level]) === "function")
      console[level](message);
  },
  autoVacuum: true
};
var defaultSearchOptions = {
  combineWith: OR,
  prefix: false,
  fuzzy: false,
  maxFuzzy: 6,
  boost: {},
  weights: { fuzzy: 0.45, prefix: 0.375 },
  bm25: defaultBM25params
};
var defaultAutoSuggestOptions = {
  combineWith: AND,
  prefix: (term, i, terms) => i === terms.length - 1
};
var defaultVacuumOptions = { batchSize: 1e3, batchWait: 10 };
var defaultVacuumConditions = { minDirtFactor: 0.1, minDirtCount: 20 };
var defaultAutoVacuumOptions = { ...defaultVacuumOptions, ...defaultVacuumConditions };
var assignUniqueTerm = (target, term) => {
  if (!target.includes(term))
    target.push(term);
};
var assignUniqueTerms = (target, source) => {
  for (const term of source) {
    if (!target.includes(term))
      target.push(term);
  }
};
var byScore = ({ score: a }, { score: b }) => b - a;
var createMap = () => /* @__PURE__ */ new Map();
var objectToNumericMap = (object) => {
  const map = /* @__PURE__ */ new Map();
  for (const key of Object.keys(object)) {
    map.set(parseInt(key, 10), object[key]);
  }
  return map;
};
var objectToNumericMapAsync = async (object) => {
  const map = /* @__PURE__ */ new Map();
  let count = 0;
  for (const key of Object.keys(object)) {
    map.set(parseInt(key, 10), object[key]);
    if (++count % 1e3 === 0) {
      await wait(0);
    }
  }
  return map;
};
var wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
var SPACE_OR_PUNCTUATION = /[\n\r\p{Z}\p{P}]+/u;

// ../../packages/search-minisearch/dist/index.js
function prepareOptions(options) {
  return {
    ...options,
    tokenize: MiniSearch.getDefault("tokenize"),
    processTerm: MiniSearch.getDefault("processTerm"),
    extractField
  };
}
function combineResults(results) {
  const [firstEntry, ...restEntries] = results.entries();
  if (firstEntry === void 0) {
    return [];
  }
  const firstRawResults = firstEntry[1];
  const initialValue = new Map(Array.from(firstRawResults.entries(), ([id, rawResult]) => {
    const { id: _, score, terms, queryTerms, match, ...rest } = rawResult;
    return [
      id,
      {
        id,
        queries: [
          {
            term: queryTerms[0],
            matches: match
          }
        ],
        ...rest
      }
    ];
  }));
  const mergedResults = restEntries.reduce((accumulator, value) => {
    const nextAccumulator = /* @__PURE__ */ new Map();
    const rawResults = value[1];
    rawResults.forEach((rawResult, docID) => {
      const existing = accumulator.get(docID);
      if (existing == null) {
        return;
      }
      const { queryTerms, match } = rawResult;
      existing.queries.push({
        term: queryTerms[0],
        matches: match
      });
      nextAccumulator.set(docID, existing);
    });
    return nextAccumulator;
  }, initialValue);
  return Array.from(mergedResults.values());
}
function createSearch(documents, options) {
  const extendedOptions = prepareOptions(options);
  const search = new MiniSearch(extendedOptions);
  search.addAll(documents.map((doc, index) => ({ ...doc, id: index })));
  return async (query) => {
    const terms = extendedOptions.tokenize(query).filter((token) => !!token);
    if (!terms.length) {
      return void 0;
    } else {
      const termResults = new Map(terms.map((term) => [
        term,
        new Map(search.search(term).map((doc) => [doc.id, doc]))
      ]));
      return combineResults(termResults);
    }
  };
}

// ../../packages/landing-pages/dist/SplitImageBlock.js
var import_jsx_runtime4 = __toESM(require_jsx_runtime(), 1);
var import_react2 = __toESM(require_react(), 1);

// ../../node_modules/unist-util-filter/lib/index.js
var own = {}.hasOwnProperty;
var filter = (
  /**
   * @type {(
   *  (<Tree extends Node, Check extends Test>(node: Tree, options: Options | null | undefined, test: Check | null | undefined) => import('./complex-types.js').Matches<Tree, Check>) &
   *  (<Tree extends Node, Check extends Test>(node: Tree, test: Check) => import('./complex-types.js').Matches<Tree, Check>) &
   *  (<Tree extends Node>(node: Tree, options?: Options | null | undefined) => Tree)
   * )}
   */
  /**
   * @param {Node} tree
   * @param {Options | Test | null | undefined} [options]
   * @param {Test | null | undefined} [test]
   * @returns {Node | null}
   */
  function(tree, options, test) {
    const is = convert(test || options);
    const cascadeRaw = options && typeof options === "object" && "cascade" in options ? (
      /** @type {boolean | null | undefined} */
      options.cascade
    ) : void 0;
    const cascade = cascadeRaw === void 0 || cascadeRaw === null ? true : cascadeRaw;
    return preorder(tree);
    function preorder(node, index, parent) {
      const children = [];
      if (!is(node, index, parent))
        return null;
      if (node.children) {
        let childIndex = -1;
        while (++childIndex < node.children.length) {
          const result = preorder(node.children[childIndex], childIndex, node);
          if (result) {
            children.push(result);
          }
        }
        if (cascade && node.children.length > 0 && children.length === 0)
          return null;
      }
      const next = {};
      let key;
      for (key in node) {
        if (own.call(node, key)) {
          next[key] = key === "children" ? children : node[key];
        }
      }
      return next;
    }
  }
);

// ../../packages/landing-pages/dist/InvalidBlock.js
var import_jsx_runtime2 = __toESM(require_jsx_runtime(), 1);

// ../../packages/landing-pages/dist/LandingBlock.js
var import_jsx_runtime = __toESM(require_jsx_runtime(), 1);
var import_classnames = __toESM(require_classnames(), 1);
function LandingBlock({ node, className, children }) {
  const grid = useGridSystemProvider();
  const { key } = node;
  const externalClassNames = (0, import_classnames.default)(className, node.class);
  const subGrid = node.visibility === "hide" ? "" : `${grid} subgrid-gap col-page [&>*]:col-page`;
  return (0, import_jsx_runtime.jsx)("div", { id: key, className: (0, import_classnames.default)("myst-landing-block relative group/block py-6", externalClassNames, {
    // Hide the subgrid if either the dataClass or the className exists and includes `col-`
    [subGrid]: !(externalClassNames && externalClassNames.includes("col-")),
    hidden: node.visibility === "remove"
  }), children }, `block-${key}`);
}

// ../../packages/landing-pages/dist/InvalidBlock.js
function InvalidBlock(props) {
  const { node, blockName } = props;
  return (0, import_jsx_runtime2.jsx)(LandingBlock, Object.assign({}, props, { children: (0, import_jsx_runtime2.jsxs)("div", { className: "myst-landing-invalid relative", role: "alert", children: [(0, import_jsx_runtime2.jsxs)("div", { className: "myst-landing-invalid-header px-4 py-2 font-bold text-white bg-red-500 rounded-t", children: ["Invalid block ", (0, import_jsx_runtime2.jsx)("span", { className: "font-mono", children: blockName })] }), (0, import_jsx_runtime2.jsxs)("div", { className: "myst-landing-invalid-border border border-t-0 border-red-400 rounded-b ", children: [(0, import_jsx_runtime2.jsx)("div", { className: "myst-landing-invalid-message px-4 py-3 text-red-700 bg-red-100", children: (0, import_jsx_runtime2.jsxs)("p", { children: ["This '", blockName, "' block does not conform to the expected AST structure."] }) }), (0, import_jsx_runtime2.jsx)("div", { className: "myst-landing-invalid-content px-4 py-3", children: (0, import_jsx_runtime2.jsx)(MyST, { ast: node.children }) })] })] }) }));
}

// ../../packages/landing-pages/dist/BlockHeading.js
var import_jsx_runtime3 = __toESM(require_jsx_runtime(), 1);
var import_react = __toESM(require_react(), 1);
var import_classnames2 = __toESM(require_classnames(), 1);
function BlockHeading({ node, className }) {
  const { enumerator, depth, key, identifier, html_id } = node;
  const id = html_id || identifier || key;
  return (0, import_react.createElement)(`h${depth}`, {
    className: (0, import_classnames2.default)("myst-landing-heading", node.class, className, "group"),
    id
  }, (0, import_jsx_runtime3.jsxs)(import_jsx_runtime3.Fragment, { children: [enumerator && (0, import_jsx_runtime3.jsx)("span", { className: "myst-landing-enumerator mr-3 select-none", children: enumerator }), (0, import_jsx_runtime3.jsx)("span", { className: "myst-landing-text heading-text", children: (0, import_jsx_runtime3.jsx)(MyST, { ast: node.children }) }), (0, import_jsx_runtime3.jsx)(HashLink, { className: "myst-landing-link font-normal", id, kind: "Section", hover: true, hideInPopup: true, noWidth: true })] }));
}

// ../../packages/landing-pages/dist/utils.js
function splitByHeader(mdast, depth) {
  let i;
  for (i = 0; i < mdast.children.length; i++) {
    const node = mdast.children[i];
    if (node.type === "heading" && (depth === void 0 || node.depth === depth)) {
      break;
    }
  }
  if (i === mdast.children.length) {
    return {
      head: void 0,
      body: { type: "block", children: mdast.children }
    };
  } else {
    return {
      head: { type: "block", children: mdast.children.slice(0, i + 1) },
      body: { type: "block", children: mdast.children.slice(i + 1) }
    };
  }
}

// ../../packages/landing-pages/dist/SplitImageBlock.js
function SplitImageBlock(props) {
  const { node } = props;
  const { image, body, links: links2, subtitle, heading } = (0, import_react2.useMemo)(() => {
    var _a, _b;
    const { head, body: rawBody } = splitByHeader(node);
    const linksNode = selectAll("link,crossReference", rawBody);
    const subtitleNode = select("paragraph", head);
    const headingNode = select("heading", head);
    const imageNode = select("image", rawBody);
    const bodyNodes = (_b = (_a = filter(rawBody, (otherNode) => !matches("link[class*=button], crossReference[class*=button], image", otherNode))) === null || _a === void 0 ? void 0 : _a.children) !== null && _b !== void 0 ? _b : [];
    return {
      body: bodyNodes,
      image: imageNode,
      links: linksNode,
      subtitle: subtitleNode,
      heading: headingNode
    };
  }, [node]);
  if (!image || !body) {
    return (0, import_jsx_runtime4.jsx)(InvalidBlock, Object.assign({}, props, { blockName: "split-image" }));
  }
  return (0, import_jsx_runtime4.jsx)(LandingBlock, Object.assign({}, props, { children: (0, import_jsx_runtime4.jsxs)("div", { className: "myst-landing-split-img relative rounded-md bg-stone-900 dark:bg-stone-800", children: [(0, import_jsx_runtime4.jsx)("div", { className: "myst-landing-split-img-image lg:absolute lg:h-full lg:w-[calc(50%)] md:absolute md:h-full md:w-[calc(100%/3)] h-80 relative [&_img]:h-full [&_img]:w-full [&_img]:object-cover [&_img]:m-0 [&_picture]:m-0 [&_picture]:inline", children: (0, import_jsx_runtime4.jsx)(MyST, { ast: image }) }), (0, import_jsx_runtime4.jsx)("div", { className: "myst-landing-split-img-content-wrapper relative py-24", children: (0, import_jsx_runtime4.jsxs)("div", { className: "myst-landing-split-img-content lg:ml-auto lg:w-[calc(50%)] lg:p-8 lg:pl-24 md:ml-auto md:w-[calc(2*100%/3)] md:pl-16 md:p-8 px-6", children: [subtitle && (0, import_jsx_runtime4.jsx)("p", { className: "myst-landing-split-img-subtitle my-0 font-semibold prose text-indigo-400 uppercase  prose-invert", children: (0, import_jsx_runtime4.jsx)(MyST, { ast: subtitle.children }) }), heading && (0, import_jsx_runtime4.jsx)(BlockHeading, { node: heading, className: "myst-landing-split-img-heading mt-2 mb-0 text-5xl font-semibold tracking-tight text-white" }), (0, import_jsx_runtime4.jsx)("div", { className: "myst-landing-split-img-body mt-6", children: (0, import_jsx_runtime4.jsx)(MyST, { ast: body, className: "prose prose-invert" }) }), links2 && (0, import_jsx_runtime4.jsx)("div", { className: "myst-landing-split-img-links flex flex-row flex-wrap items-center gap-4 mt-8", children: (0, import_jsx_runtime4.jsx)(MyST, { ast: links2, className: "prose prose-invert" }) })] }) })] }) }));
}
var SPLIT_IMAGE_RENDERERS = {
  block: {
    "block[kind=split-image]": SplitImageBlock
  }
};

// ../../packages/landing-pages/dist/CenteredBlock.js
var import_jsx_runtime5 = __toESM(require_jsx_runtime(), 1);
var import_react3 = __toESM(require_react(), 1);
function CenteredBlock(props) {
  const { node } = props;
  const { body, links: links2, subtitle, heading } = (0, import_react3.useMemo)(() => {
    var _a, _b;
    const { head, body: rawBody } = splitByHeader(node);
    const linksNode = selectAll("link[class*=button], crossReference[class*=button]", rawBody);
    const subtitleNode = select("paragraph", head);
    const headingNode = select("heading", head);
    const bodyNodes = (_b = (_a = filter(rawBody, (otherNode) => !matches("link[class*=button], crossReference[class*=button]", otherNode))) === null || _a === void 0 ? void 0 : _a.children) !== null && _b !== void 0 ? _b : [];
    return {
      body: bodyNodes,
      links: linksNode,
      subtitle: subtitleNode,
      heading: headingNode
    };
  }, [node]);
  if (!body) {
    return (0, import_jsx_runtime5.jsx)(InvalidBlock, Object.assign({}, props, { blockName: "centered" }));
  }
  return (0, import_jsx_runtime5.jsx)(LandingBlock, Object.assign({}, props, { children: (0, import_jsx_runtime5.jsx)("div", { className: "myst-landing-centered relative text-center", children: (0, import_jsx_runtime5.jsxs)("div", { className: "myst-landing-centered-inner py-20 sm:py-28", children: [subtitle && (0, import_jsx_runtime5.jsx)("p", { className: "myst-landing-centered-subtitle my-0 font-semibold text-indigo-400 uppercase", children: (0, import_jsx_runtime5.jsx)(MyST, { ast: subtitle.children }) }), heading && (0, import_jsx_runtime5.jsx)(BlockHeading, { node: heading, className: "myst-landing-centered-heading mt-2 mb-0 text-5xl font-semibold tracking-tight" }), body && (0, import_jsx_runtime5.jsx)("div", { className: "myst-landing-centered-body mt-6", children: (0, import_jsx_runtime5.jsx)(MyST, { ast: body }) }), links2 && (0, import_jsx_runtime5.jsx)("div", { className: "myst-landing-centered-links flex flex-row flex-wrap items-center justify-center gap-4 mt-8", children: (0, import_jsx_runtime5.jsx)(MyST, { ast: links2 }) })] }) }) }));
}
var CENTERED_RENDERERS = {
  block: {
    "block[kind=centered]": CenteredBlock
  }
};

// ../../packages/landing-pages/dist/JustifiedBlock.js
var import_jsx_runtime6 = __toESM(require_jsx_runtime(), 1);
var import_react4 = __toESM(require_react(), 1);
function JustifiedBlock(props) {
  const { node } = props;
  const { body, links: links2, subtitle, heading } = (0, import_react4.useMemo)(() => {
    var _a, _b;
    const { head, body: rawBody } = splitByHeader(node);
    const linksNode = selectAll("link[class*=button], crossReference[class*=button]", rawBody);
    const subtitleNode = select("paragraph", head);
    const headingNode = select("heading", head);
    const bodyNodes = (_b = (_a = filter(rawBody, (otherNode) => !matches("link[class*=button], crossReference[class*=button]", otherNode))) === null || _a === void 0 ? void 0 : _a.children) !== null && _b !== void 0 ? _b : [];
    return {
      body: bodyNodes,
      links: linksNode,
      subtitle: subtitleNode,
      heading: headingNode
    };
  }, [node]);
  if (!body) {
    return (0, import_jsx_runtime6.jsx)(InvalidBlock, Object.assign({}, props, { blockName: "justified" }));
  }
  return (0, import_jsx_runtime6.jsx)(LandingBlock, Object.assign({}, props, { children: (0, import_jsx_runtime6.jsxs)("div", { className: "myst-landing-justified grid-areas-just-narrow md:grid-areas-just-wide grid grid-cols-[auto_1fr_auto] justify-between gap-2 py-20 sm:py-28 lg:px-8", children: [subtitle && (0, import_jsx_runtime6.jsx)("p", { className: "myst-landing-justified-subtitle grid-area-subtitle my-0 font-semibold text-indigo-400 uppercase", children: (0, import_jsx_runtime6.jsx)(MyST, { ast: subtitle.children }) }), heading && (0, import_jsx_runtime6.jsx)(BlockHeading, { node: heading, className: "myst-landing-justified-heading grid-area-title mt-2 mb-0 text-5xl font-semibold tracking-tight" }), body && (0, import_jsx_runtime6.jsx)("div", { className: "myst-landing-justified-body grid-area-body mt-6", children: (0, import_jsx_runtime6.jsx)(MyST, { ast: body }) }), links2 && (0, import_jsx_runtime6.jsx)("div", { className: "myst-landing-justified-links grid-area-links flex flex-row flex-wrap items-center gap-4", children: (0, import_jsx_runtime6.jsx)(MyST, { ast: links2 }) })] }) }));
}
var JUSTIFIED_RENDERERS = {
  block: {
    "block[kind=justified]": JustifiedBlock
  }
};

// ../../packages/landing-pages/dist/LogoCloudBlock.js
var import_jsx_runtime7 = __toESM(require_jsx_runtime(), 1);
var import_react5 = __toESM(require_react(), 1);
function LogoCloudBlock(props) {
  const { node } = props;
  const { grid, body, links: links2 } = (0, import_react5.useMemo)(() => {
    var _a, _b;
    const gridNode = select("grid", node);
    const rawBodyNode = filter(node, (child) => child.type !== "grid");
    const linksNode = selectAll("link[class*=button], crossReference[class*=button]", rawBodyNode);
    const bodyNodes = (_b = (_a = filter(rawBodyNode, (otherNode) => !matches("link[class*=button], crossReference[class*=button]", otherNode))) === null || _a === void 0 ? void 0 : _a.children) !== null && _b !== void 0 ? _b : [];
    return {
      body: bodyNodes,
      grid: gridNode,
      links: linksNode
    };
  }, [node]);
  if (!grid) {
    return (0, import_jsx_runtime7.jsx)(InvalidBlock, Object.assign({}, props, { blockName: "logo-cloud" }));
  }
  return (0, import_jsx_runtime7.jsx)(LandingBlock, Object.assign({}, props, { children: (0, import_jsx_runtime7.jsxs)("div", { className: "myst-landing-logo-cloud py-20 text-center sm:py-28", children: [(0, import_jsx_runtime7.jsx)("div", { className: "myst-landing-logo-cloud-body font-semibold", children: (0, import_jsx_runtime7.jsx)(MyST, { ast: body }) }), grid && (0, import_jsx_runtime7.jsx)(MyST, { ast: grid }), links2 && (0, import_jsx_runtime7.jsx)("div", { className: "myst-landing-logo-cloud-links flex flex-row flex-wrap items-center justify-center gap-4 mt-8", children: (0, import_jsx_runtime7.jsx)(MyST, { ast: links2 }) })] }) }));
}
var LOGO_CLOUD_RENDERERS = {
  block: {
    "block[kind=logo-cloud]": LogoCloudBlock
  }
};

// ../../packages/landing-pages/dist/index.js
var LANDING_PAGE_RENDERERS = mergeRenderers([
  SPLIT_IMAGE_RENDERERS,
  CENTERED_RENDERERS,
  JUSTIFIED_RENDERERS,
  LOGO_CLOUD_RENDERERS
]);

// ../../packages/anywidget/dist/renderers.js
var import_jsx_runtime8 = __toESM(require_jsx_runtime(), 1);
var React = __toESM(require_react(), 1);
var import_classnames3 = __toESM(require_classnames(), 1);

// ../../packages/anywidget/dist/models.js
var __classPrivateFieldSet = function(receiver, state, value, kind, f) {
  if (kind === "m")
    throw new TypeError("Private method is not writable");
  if (kind === "a" && !f)
    throw new TypeError("Private accessor was defined without a setter");
  if (typeof state === "function" ? receiver !== state || !f : !state.has(receiver))
    throw new TypeError("Cannot write private member to an object whose class did not declare it");
  return kind === "a" ? f.call(receiver, value) : f ? f.value = value : state.set(receiver, value), value;
};
var __classPrivateFieldGet = function(receiver, state, kind, f) {
  if (kind === "a" && !f)
    throw new TypeError("Private accessor was defined without a getter");
  if (typeof state === "function" ? receiver !== state || !f : !state.has(receiver))
    throw new TypeError("Cannot read private member from an object whose class did not declare it");
  return kind === "m" ? f : kind === "a" ? f.call(receiver) : f ? f.value : state.get(receiver);
};
var _MystAnyModel_state;
var _MystAnyModel_target;
var MystAnyModel = class {
  constructor(state) {
    _MystAnyModel_state.set(this, void 0);
    _MystAnyModel_target.set(this, new EventTarget());
    __classPrivateFieldSet(this, _MystAnyModel_state, state, "f");
  }
  get(name) {
    return __classPrivateFieldGet(this, _MystAnyModel_state, "f")[name];
  }
  set(key, value) {
    __classPrivateFieldGet(this, _MystAnyModel_state, "f")[key] = value;
    __classPrivateFieldGet(this, _MystAnyModel_target, "f").dispatchEvent(new CustomEvent(`change:${key}`, { detail: value }));
    __classPrivateFieldGet(this, _MystAnyModel_target, "f").dispatchEvent(new CustomEvent("change", { detail: value }));
  }
  on(name, cb) {
    __classPrivateFieldGet(this, _MystAnyModel_target, "f").addEventListener(name, cb);
  }
  off(_name, _cb) {
    throw new Error("MystAnyModel.off not implemented yet.");
  }
  save_changes() {
    throw new Error("MystAnyModel.save_changes not implemented yet.");
  }
  send(_msg, _callbacks, _buffers) {
    throw new Error("MystAnyModel.send not implemented yet.");
  }
  get widget_manager() {
    throw new Error("MystAnyModel.widget_manager does not exist.");
  }
};
_MystAnyModel_state = /* @__PURE__ */ new WeakMap(), _MystAnyModel_target = /* @__PURE__ */ new WeakMap();

// ../../packages/anywidget/dist/renderers.js
var __awaiter = function(thisArg, _arguments, P, generator) {
  function adopt(value) {
    return value instanceof P ? value : new P(function(resolve) {
      resolve(value);
    });
  }
  return new (P || (P = Promise))(function(resolve, reject) {
    function fulfilled(value) {
      try {
        step(generator.next(value));
      } catch (e2) {
        reject(e2);
      }
    }
    function rejected(value) {
      try {
        step(generator["throw"](value));
      } catch (e2) {
        reject(e2);
      }
    }
    function step(result) {
      result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected);
    }
    step((generator = generator.apply(thisArg, _arguments || [])).next());
  });
};
function AnyWidgetRenderer({ node }) {
  var _a;
  const esmModuleUrl = node.esm;
  const isESMModuleUrlValid = esmModuleUrl && typeof esmModuleUrl === "string";
  const validModel = node.model && typeof node.model === "object";
  const ref = React.useRef(null);
  const [error, setError] = React.useState(null);
  React.useEffect(() => {
    setError(null);
    const controller = new AbortController();
    if (controller.signal.aborted) {
      return;
    }
    let maybeCleanupInitialize = void 0;
    let maybeCleanupRender = void 0;
    controller.signal.addEventListener("abort", () => __awaiter(this, void 0, void 0, function* () {
      yield maybeCleanupRender === null || maybeCleanupRender === void 0 ? void 0 : maybeCleanupRender();
      yield maybeCleanupInitialize === null || maybeCleanupInitialize === void 0 ? void 0 : maybeCleanupInitialize();
    }));
    const useShadowDom = true;
    console.debug("AnyRenderer importing:", esmModuleUrl);
    import(esmModuleUrl).then((mod) => __awaiter(this, void 0, void 0, function* () {
      var _a2, _b, _c;
      const rootEl = ref.current;
      if (!rootEl)
        return;
      console.debug("AnyRenderer imported", mod);
      const widget = mod.default;
      const model = new MystAnyModel(node.model);
      maybeCleanupInitialize = yield (_a2 = widget.initialize) === null || _a2 === void 0 ? void 0 : _a2.call(widget, { model });
      rootEl.className = (0, import_classnames3.default)("myst-anywidget", node.class);
      let widgetRoot;
      if (useShadowDom) {
        const shadowRoot = (_b = rootEl.shadowRoot) !== null && _b !== void 0 ? _b : rootEl.attachShadow({ mode: "open" });
        widgetRoot = document.createElement("div");
        widgetRoot.className = (0, import_classnames3.default)("myst-anywidget", node.class);
        widgetRoot.style.position = "relative";
        shadowRoot.replaceChildren(widgetRoot);
      } else {
        widgetRoot = rootEl;
      }
      if (node.css) {
        const link = document.createElement("link");
        link.rel = "stylesheet";
        link.href = node.css;
        widgetRoot.appendChild(link);
      }
      maybeCleanupRender = yield (_c = widget.render) === null || _c === void 0 ? void 0 : _c.call(widget, {
        model,
        el: widgetRoot
      });
    })).catch((err) => {
      console.error("AnyRenderer failed to import module:", err);
      setError(err instanceof Error ? err : new Error(String(err)));
    });
    return () => {
      controller === null || controller === void 0 ? void 0 : controller.abort();
    };
  }, [node]);
  if (error) {
    return (0, import_jsx_runtime8.jsxs)("details", { className: "p-3 bg-gray-100 rounded border border-gray-300 cursor-pointer", children: [(0, import_jsx_runtime8.jsxs)("summary", { className: "text-sm text-gray-600 select-none", children: ["Failed to load ", (0, import_jsx_runtime8.jsx)("code", { className: "text-xs", children: "anywidget" }), " module."] }), (0, import_jsx_runtime8.jsxs)("div", { className: "pt-2 mt-2 space-y-1 text-xs border-t border-gray-200", children: [(0, import_jsx_runtime8.jsxs)("div", { className: "text-gray-500", children: [(0, import_jsx_runtime8.jsx)("span", { className: "font-medium", children: "Widget Module URL:" }), " ", esmModuleUrl] }), (0, import_jsx_runtime8.jsxs)("div", { className: "text-gray-700", children: [(0, import_jsx_runtime8.jsx)("span", { className: "font-medium", children: "Error:" }), " ", error.message] }), (0, import_jsx_runtime8.jsxs)("div", { className: "text-gray-700", children: [(0, import_jsx_runtime8.jsx)("span", { className: "font-medium", children: "Stack:" }), " ", error.stack] })] })] });
  }
  if (!isESMModuleUrlValid || !validModel) {
    return (0, import_jsx_runtime8.jsxs)("div", { className: "p-3 space-y-2 rounded-md border border-red-500", children: [(0, import_jsx_runtime8.jsxs)("div", { children: ["Invalid ", (0, import_jsx_runtime8.jsx)("code", { children: "anywidget" }), " directive."] }), !isESMModuleUrlValid && (0, import_jsx_runtime8.jsxs)("div", { className: "px-1", children: [(0, import_jsx_runtime8.jsx)("div", { children: "Invalid import URL" }), (0, import_jsx_runtime8.jsx)("div", { className: "text-sm text-gray-500", children: node.esm })] }), !validModel && (0, import_jsx_runtime8.jsxs)("div", { className: "px-1", children: [(0, import_jsx_runtime8.jsx)("div", { children: "Invalid JSON data" }), (0, import_jsx_runtime8.jsx)("div", { className: "text-sm text-gray-500", children: (_a = node.model) === null || _a === void 0 ? void 0 : _a.toString() })] })] });
  }
  return (0, import_jsx_runtime8.jsx)("div", { className: "relative w-full", ref });
}

// ../../packages/anywidget/dist/index.js
var ANY_RENDERERS = {
  anywidget: AnyWidgetRenderer
};

// app/root.tsx
var import_react7 = __toESM(require_react());
var import_jsx_dev_runtime = __toESM(require_jsx_dev_runtime());
var RENDERERS = mergeRenderers(
  [
    renderers,
    JUPYTER_RENDERERS,
    LANDING_PAGE_RENDERERS,
    ANY_RENDERERS
  ]
);
var meta = ({ data }) => {
  var _a, _b, _c, _d;
  return getMetaTagsForSite({
    title: (_a = data == null ? void 0 : data.config) == null ? void 0 : _a.title,
    description: (_b = data == null ? void 0 : data.config) == null ? void 0 : _b.description,
    twitter: (_d = (_c = data == null ? void 0 : data.config) == null ? void 0 : _c.options) == null ? void 0 : _d.twitter
  });
};
var links = () => {
  return [
    { rel: "stylesheet", href: app_default },
    { rel: "stylesheet", href: thebe_core_default },
    {
      rel: "stylesheet",
      href: "https://cdn.jsdelivr.net/npm/jupyter-matplotlib@0.11.3/css/mpl_widget.css"
    },
    {
      rel: "stylesheet",
      href: "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.css"
    }
  ];
};
function createSearch2(index) {
  const options = {
    fields: SEARCH_ATTRIBUTES_ORDERED,
    storeFields: ["hierarchy", "content", "url", "type", "id", "position"],
    idField: "id",
    searchOptions: {
      fuzzy: 0.2,
      prefix: true
    }
  };
  return createSearch(index.records, options);
}
function NoCSSWarning() {
  const CLIENT_THEME_SOURCE = `
    (() => {
            // Test for has-styling variable set by the MyST stylesheet
            const node = document.currentScript.parentNode;
            const hasCSS = window.getComputedStyle(node).getPropertyValue("--has-styling");
            if (hasCSS === ""){
                    node.showModal();
            }

    })()
`;
  return /* @__PURE__ */ (0, import_jsx_dev_runtime.jsxDEV)(import_jsx_dev_runtime.Fragment, { children: /* @__PURE__ */ (0, import_jsx_dev_runtime.jsxDEV)(
    "dialog",
    {
      id: "myst-no-css",
      style: {
        position: "fixed",
        left: "0px",
        top: "0px",
        width: "100%",
        height: "100vh",
        fontSize: "4rem",
        padding: "1rem",
        color: "black",
        background: "white"
      },
      suppressHydrationWarning: true,
      children: [
        /* @__PURE__ */ (0, import_jsx_dev_runtime.jsxDEV)("strong", { children: "Site not loading correctly?" }, void 0, false, {
          fileName: "app/root.tsx",
          lineNumber: 124,
          columnNumber: 9
        }, this),
        /* @__PURE__ */ (0, import_jsx_dev_runtime.jsxDEV)("p", { children: [
          "This may be due to an incorrect ",
          /* @__PURE__ */ (0, import_jsx_dev_runtime.jsxDEV)("code", { children: "BASE_URL" }, void 0, false, {
            fileName: "app/root.tsx",
            lineNumber: 126,
            columnNumber: 43
          }, this),
          " configuration. See",
          " ",
          /* @__PURE__ */ (0, import_jsx_dev_runtime.jsxDEV)("a", { href: "https://mystmd.org/guide/deployment#deploy-base-url", children: "the MyST Documentation" }, void 0, false, {
            fileName: "app/root.tsx",
            lineNumber: 127,
            columnNumber: 11
          }, this),
          " ",
          "for reference."
        ] }, void 0, true, {
          fileName: "app/root.tsx",
          lineNumber: 125,
          columnNumber: 9
        }, this),
        /* @__PURE__ */ (0, import_jsx_dev_runtime.jsxDEV)("script", { dangerouslySetInnerHTML: { __html: CLIENT_THEME_SOURCE } }, void 0, false, {
          fileName: "app/root.tsx",
          lineNumber: 130,
          columnNumber: 9
        }, this)
      ]
    },
    void 0,
    true,
    {
      fileName: "app/root.tsx",
      lineNumber: 107,
      columnNumber: 7
    },
    this
  ) }, void 0, false, {
    fileName: "app/root.tsx",
    lineNumber: 106,
    columnNumber: 5
  }, this);
}
function AppWithReload() {
  const { theme, config, CONTENT_CDN_PORT, MODE, BASE_URL } = useLoaderData();
  const searchFactory = (0, import_react7.useCallback)((index) => createSearch2(index), []);
  return /* @__PURE__ */ (0, import_jsx_dev_runtime.jsxDEV)(SearchFactoryProvider, { factory: searchFactory, children: /* @__PURE__ */ (0, import_jsx_dev_runtime.jsxDEV)(
    Document,
    {
      theme,
      config,
      scripts: MODE === "static" ? void 0 : /* @__PURE__ */ (0, import_jsx_dev_runtime.jsxDEV)(ContentReload, { port: CONTENT_CDN_PORT }, void 0, false, {
        fileName: "app/root.tsx",
        lineNumber: 146,
        columnNumber: 50
      }, this),
      staticBuild: MODE === "static",
      baseurl: BASE_URL,
      renderers: RENDERERS,
      head: /* @__PURE__ */ (0, import_jsx_dev_runtime.jsxDEV)(import_jsx_dev_runtime.Fragment, { children: [
        /* @__PURE__ */ (0, import_jsx_dev_runtime.jsxDEV)("link", { rel: "icon", href: `${BASE_URL || ""}/favicon.ico` }, void 0, false, {
          fileName: "app/root.tsx",
          lineNumber: 152,
          columnNumber: 13
        }, this),
        /* @__PURE__ */ (0, import_jsx_dev_runtime.jsxDEV)("link", { rel: "stylesheet", href: `${BASE_URL || ""}/myst-theme.css` }, void 0, false, {
          fileName: "app/root.tsx",
          lineNumber: 153,
          columnNumber: 13
        }, this)
      ] }, void 0, true, {
        fileName: "app/root.tsx",
        lineNumber: 151,
        columnNumber: 9
      }, this),
      children: [
        /* @__PURE__ */ (0, import_jsx_dev_runtime.jsxDEV)(
          SkipTo,
          {
            targets: [
              { id: "skip-to-frontmatter", title: "Skip to article frontmatter" },
              { id: "skip-to-article", title: "Skip to article content" }
            ]
          },
          void 0,
          false,
          {
            fileName: "app/root.tsx",
            lineNumber: 157,
            columnNumber: 9
          },
          this
        ),
        /* @__PURE__ */ (0, import_jsx_dev_runtime.jsxDEV)(NoCSSWarning, {}, void 0, false, {
          fileName: "app/root.tsx",
          lineNumber: 163,
          columnNumber: 9
        }, this),
        /* @__PURE__ */ (0, import_jsx_dev_runtime.jsxDEV)(Outlet, {}, void 0, false, {
          fileName: "app/root.tsx",
          lineNumber: 164,
          columnNumber: 9
        }, this)
      ]
    },
    void 0,
    true,
    {
      fileName: "app/root.tsx",
      lineNumber: 143,
      columnNumber: 7
    },
    this
  ) }, void 0, false, {
    fileName: "app/root.tsx",
    lineNumber: 142,
    columnNumber: 5
  }, this);
}
export {
  AppErrorBoundary as ErrorBoundary,
  AppWithReload as default,
  links,
  meta
};
//# sourceMappingURL=/build/root-K5AZREDU.js.map
