/**
* Code borrowed from CMPUT 415 Cymbol code
*/

#include "ASTWalker.h"
#include "GazpreaParser.h"
#include "BuiltInTypeSymbol.h"

namespace gazprea {
    DefRef::DefRef(SymbolTable *symtab, AST *root) : symtab(symtab) {
        // define global scope
        std::string globals = "globals";
        currentScope = symtab->enterScope(globals, nullptr);
        currentScope->define(new BuiltInTypeSymbol("int"));
        currentScope->define(new BuiltInTypeSymbol("vector"));
        symtab->globalScope = currentScope;
        root->scope = currentScope;
    }


    std::any DefRef::visit(AST *t) {
        if (t->isNil()) {
            visitChildren(t);
        } else {
            switch (t->getNodeType()) {
                case GazpreaParser::VAR_DECL:
                    visitVAR_DECL(t);
                    break;
                case GazpreaParser::ASSIGN:
                    visitASSIGN(t);
                    break;
                case GazpreaParser::LOOP:
                case GazpreaParser::CONDITIONAL:
                    visitLOOPCONDITIONAL(t);
                    break;
                case GazpreaParser::FILTER:
                case GazpreaParser::GENERATOR:
                    visitFILTERGENERATOR(t);
                    break;
                case GazpreaParser::ID:
                    visitID(t);
                    break;
                default:
                    // The other nodes we don't care about just have their children visited
                    visitChildren(t);
            }
        }
        return 0;
    }

    void DefRef::visitChildren(AST *t) {
        for (auto child : t->children) {
            if (!child->scope) child->scope = currentScope;
            visit(child);
        }
    }

    void DefRef::visitASSIGN(AST *t) {
        AST* var = t->children[0];
        std::string idStr = var->token->getText();

        std::cout << "Assigning " << idStr << std::endl;
        // resolve id
        VariableSymbol* vs = dynamic_cast<VariableSymbol*>(currentScope->resolve(idStr));
        if (vs) {
            std::cout << "(assign): resolved " << vs->source() << std::endl;
        } else {
            std::cerr << "Assignment Error line " << t->loc()
                      << ": undefined symbol " << idStr
                      << std::endl;
        }

        var->symbol = vs;
        std::cout << "Looking at assignment expr" << std::endl;
        visit(t->children[1]);
    }

    void DefRef::visitLOOPCONDITIONAL(AST *t) {
        AST* expr = t->children[0];
        visit(expr);

        // enter conditional scope
        std::string sname = "loopcond" + std::to_string(t->loc());
        currentScope = symtab->enterScope(sname, currentScope);

        for (int i = 1; i < t->children.size(); i++) {
            visit(t->children[i]);
        }

        currentScope = symtab->exitScope(currentScope);
    }

    void DefRef::visitVAR_DECL(AST *t) {
        Type *typeAST = resolveType(t->children[0]);
        AST* id = t->children[1];
        AST* expr = t->children[2];

        assert(typeAST);

        // Visiting the children of the expression before putting the variable in scope
        visit(expr);

        // define variable
        VariableSymbol* vs = new VariableSymbol(id->token->getText(), typeAST, currentScope);
        std::cout << "VARIABLE SYMBOL TYPE:" << vs->type << std::endl;
        currentScope->define(vs);
        id->symbol = vs;

        t->scope = currentScope;

        std::cout << "VARDECL: created varSymbol " << vs->toString() << " in scope " << currentScope->toString() << std::endl;
    }

    void DefRef::visitFILTERGENERATOR(AST *t) {
        AST* id = t->children[0];
        AST* domain = t->children[1];
        AST* expression = t->children[2];

        // domain cannot see the filter identifier scope
        visit(domain);

        // enter local scope
        std::string sname = "genfilter" + std::to_string(t->loc());
        currentScope = symtab->enterScope(sname, currentScope);

        // the identifier in a filter expression must be an integer
        VariableSymbol* vs = new VariableSymbol(id->token->getText(), new BuiltInTypeSymbol("int"), currentScope);

        // define variable in inner scope
        currentScope->define(vs);
        id->symbol = vs;
        id->scope = currentScope;

        std::cout << "Entered scope " << currentScope->toString() << std::endl;
        visit(expression);

        currentScope = symtab->exitScope(currentScope);

        std::cout << "Visited generator/filter '" << sname << "'" << std::endl;
    }

    void DefRef::visitID(AST *t) {
        if (t->symbol)
            return;

        t->symbol = t->scope->resolve(t->token->getText());
        if (!t->symbol) {
            std::cerr << "Error line " << t->loc()
                      << ": undefined symbol " << t->token->getText()
                      << std::endl;
        }

        std::cout << "SYMBOL TYPE IN DEFREF: " << t->symbol->type << std::endl;
        std::cout << "resolved " << t->symbol->source();
    }


    BuiltInTypeSymbol* DefRef::resolveType(AST *t) {
        std::string tokenStr = t->token->getText();
        std::cout << "Resolve Type: " << tokenStr << std::endl;
        if (tokenStr == "int") {
            return dynamic_cast<BuiltInTypeSymbol*>(symtab->globalScope->resolve("int"));
        }
        else if (tokenStr == "vector") {
            return dynamic_cast<BuiltInTypeSymbol*>(symtab->globalScope->resolve("vector"));
        }

        // this should not happen...
        std::cerr << "Unknown type: " << tokenStr << "!" << std::endl;
        return nullptr;
    }
}
