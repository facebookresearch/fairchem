import {
  require_markup_templating
} from "/build/_shared/chunk-E6QZW6U4.js";
import {
  require_ruby
} from "/build/_shared/chunk-IZ2CIBO4.js";
import {
  __commonJS
} from "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/refractor/lang/erb.js
var require_erb = __commonJS({
  "../../node_modules/refractor/lang/erb.js"(exports, module) {
    var refractorRuby = require_ruby();
    var refractorMarkupTemplating = require_markup_templating();
    module.exports = erb;
    erb.displayName = "erb";
    erb.aliases = [];
    function erb(Prism) {
      Prism.register(refractorRuby);
      Prism.register(refractorMarkupTemplating);
      (function(Prism2) {
        Prism2.languages.erb = {
          delimiter: {
            pattern: /^(\s*)<%=?|%>(?=\s*$)/,
            lookbehind: true,
            alias: "punctuation"
          },
          ruby: {
            pattern: /\s*\S[\s\S]*/,
            alias: "language-ruby",
            inside: Prism2.languages.ruby
          }
        };
        Prism2.hooks.add("before-tokenize", function(env) {
          var erbPattern = /<%=?(?:[^\r\n]|[\r\n](?!=begin)|[\r\n]=begin\s(?:[^\r\n]|[\r\n](?!=end))*[\r\n]=end)+?%>/g;
          Prism2.languages["markup-templating"].buildPlaceholders(
            env,
            "erb",
            erbPattern
          );
        });
        Prism2.hooks.add("after-tokenize", function(env) {
          Prism2.languages["markup-templating"].tokenizePlaceholders(env, "erb");
        });
      })(Prism);
    }
  }
});

export {
  require_erb
};
//# sourceMappingURL=/build/_shared/chunk-O7T3B7VZ.js.map
